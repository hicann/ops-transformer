/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_INT8_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_INT8_H_

#include "kernel_operator.h"
#include "quant_flash_attn_tiling_data_int8.h"
#include "quant_flash_attn_common_int8.h"

namespace BaseApi {

using namespace AscendC;

// AIV subcores own disjoint Q halves and retain online state in UB across KV tiles. GM slots connect MM1/V1/MM2/V2.

template <bool ENABLE>
struct QuantFlashAttnLseState {};
template <>
struct QuantFlashAttnLseState<true> {
    AscendC::GlobalTensor<float> lseGm_;
};

template <bool RETURN_LSE, uint32_t STATIC_D = 0U, uint32_t STATIC_M = 0U, Int8GqaPath PATH = Int8GqaPath::BNSD,
          bool D96_PREFILL = false, uint16_t D64_VEC_FEATURES = 0U, bool D64_REUSE_MASK = false,
          bool K_SCALE_PREFIX_CACHE = false, bool WINDOWED = false>
class QuantFlashAttnBlockVecGqaInt8 : private QuantFlashAttnLseState<RETURN_LSE> {
public:
    __aicore__ inline void Init(__gm__ uint8_t* workspaceCore, __gm__ uint8_t* out, __gm__ uint8_t* qDescale,
                                __gm__ uint8_t* kDescale, __gm__ uint8_t* vDescale, __gm__ uint8_t* pScale,
                                __gm__ uint8_t* softmaxLse, const QuantFlashAttnTilingDataInt8& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        if constexpr (RETURN_LSE)
            this->lseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmaxLse));
        // Keep the UB policy fixed for the call: packed members may later rebind a shorter KV sequence without
        // changing allocated buffers.
        if constexpr (STATIC_D == 0U && STATIC_M == 0U) {
            largeFp32Accumulator_ = tiling_.blockM == Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE &&
                                    tiling_.headDimAlign > Int8GqaTnd::QFA_LARGE_FP32_HEAD_DIM_THRESHOLD &&
                                    tiling_.groupSize > 1U && tiling_.seqLenKv > tiling_.blockN;
            partitionedMhaHalf_ = tiling_.groupSize == 1U &&
                                  tiling_.seqLenQ > MHA_PARTITIONED_ACCUMULATOR_QUERY_THRESHOLD &&
                                  tiling_.blockM < Int8GqaTnd::QFA_LARGE_QUERY_TILE_ROWS &&
                                  tiling_.headDimAlign <= Int8GqaTnd::QFA_LARGE_FP32_HEAD_DIM_THRESHOLD &&
                                  tiling_.seqLenKv > tiling_.blockN;
        }
        if constexpr (PATH != Int8GqaPath::BNSD) {
            cacheK_ = QuantFlashAttnLayoutInt8<PATH>::QfaUseKScaleCache(tiling_);
        }
        if constexpr (K_SCALE_PREFIX_CACHE)
            cacheK_ = true;

        subBlock_ = GetBlockIdx() & 1U;
        overlayPv_ = (BlockM() == Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE ||
                      BlockM() == Int8GqaTnd::QFA_LARGE_QUERY_TILE_ROWS);
        slotCount_ = BlockM() >= Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE ? Int8GqaTnd::QFA_PV_OVERLAY_SLOTS :
                                                                             Int8GqaTnd::QFA_PIPELINE_SLOTS;
        const uint32_t scoreBytes = slotCount_ * tiling_.scoreSlotElems * sizeof(half);
        const uint32_t pBytes = slotCount_ * tiling_.pSlotElems * sizeof(int8_t);
        scoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceCore));
        pGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspaceCore + scoreBytes));
        pvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceCore + scoreBytes + (overlayPv_ ? 0U : pBytes)));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
        qDescaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(qDescale));
        vDescaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vDescale));
        kDescaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kDescale));
        kDeqWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
            workspaceCore + tiling_.coreWorkspaceBytes -
            Int8GqaTnd::QfaKScaleWorkspaceElements(cacheK_, K_SCALE_PREFIX_CACHE, slotCount_, tiling_.blockN) *
                sizeof(uint64_t)));

        GlobalTensor<float> pScaleGm;
        pScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pScale));
        pScaleValue_ = pScaleGm.GetValue(0);

        // Vec1 scores and Vec2 PV never coexist in UB. P8 and the Vec2 FP32 PV cast also have disjoint lifetimes.
        pipe->InitBuffer(dataBuf_, VecRows() * MAX_BLOCK_N * sizeof(float));
        if constexpr (D96_PREFILL) {
            // The D96 prefill entries accumulate M512/D96 as FP16.
            pipe->InitBuffer(accBuf_, (STATIC_M / Int8GqaTnd::QFA_VECTOR_CORES_PER_CUBE) * Int8GqaTnd::QFA_HEAD_DIM_96 *
                                          sizeof(half));
        } else {
            pipe->InitBuffer(accBuf_, largeFp32Accumulator_ ?
                                          Int8GqaTnd::QFA_LARGE_FP32_ACCUMULATOR_BYTES :
                                          MAX_ACC_ELEMS * (STATIC_D != 0U && STATIC_D <= Int8GqaTnd::QFA_HEAD_DIM_64 ?
                                                               sizeof(half) :
                                                               sizeof(float)));
        }
        pipe->InitBuffer(tmpBuf_, TmpBytes());
        pipe->InitBuffer(maxBuf_, ONLINE_STATE_BUFFER_COUNT * MAX_TASK_ROWS * sizeof(float));
        pipe->InitBuffer(sumBuf_, ONLINE_STATE_BUFFER_COUNT * MAX_TASK_ROWS * sizeof(float));
        pipe->InitBuffer(coeffBuf_, SLOT_COUNT * MAX_TASK_ROWS * sizeof(float));
        pipe->InitBuffer(brcBuf_, MAX_BROADCAST_ROWS * QFA_INT8_DMA_BLOCK_BYTES);
        pipe->InitBuffer(quantScaleBuf_, VECTOR_REPEAT_BYTES);
        pipe->InitBuffer(rowMaxBuf_, MAX_SEGMENTS * MAX_REDUCE_ROWS * sizeof(half));
        pipe->InitBuffer(rowSumBuf_, MAX_SEGMENTS * MAX_REDUCE_ROWS * sizeof(half));
        pipe->InitBuffer(rowBrcBuf_, VecRows() * QFA_INT8_DMA_BLOCK_BYTES);
        pipe->InitBuffer(rowMaxFloatBuf_, VecRows() * sizeof(float));
        pipe->InitBuffer(rowSumFloatBuf_, MAX_SEGMENTS * MAX_REDUCE_ROWS * sizeof(float));

        // Immutable Fixpipe packing indices must survive P/PV scratch reuse.
        pipe->InitBuffer(qScaleBuf_, MAX_TASK_ROWS * (sizeof(half) + sizeof(int32_t)));
        pipe->InitBuffer(kScaleIndexBuf_, MAX_BLOCK_N * sizeof(uint32_t));
        LocalTensor<int32_t> index = kScaleIndexBuf_.Get<int32_t>();
        CreateVecIndex(index, 0, MAX_BLOCK_N);
        PipeBarrier<PIPE_V>();

        {
            auto qIndex = qScaleBuf_.Get<uint8_t>()[MAX_TASK_ROWS * sizeof(half)].ReinterpretCast<int32_t>();
            Adds(qIndex, index, 0, MAX_TASK_ROWS);
            PipeBarrier<PIPE_V>();
        }
        ShiftRight(index.ReinterpretCast<uint32_t>(), index.ReinterpretCast<uint32_t>(), 1U, MAX_BLOCK_N);
        PipeBarrier<PIPE_V>();
        Muls(index, index, static_cast<int32_t>(sizeof(float)), MAX_BLOCK_N);
        PipeBarrier<PIPE_V>();

        scoreUb_ = dataBuf_.Get<half>();
        pvUb_ = dataBuf_.Get<half>();
        pUb_ = tmpBuf_.Get<int8_t>();
        maxStateUb_ = maxBuf_.Get<float>();
        sumStateUb_ = sumBuf_.Get<float>();
        coeffUb_ = coeffBuf_.Get<float>();
        brcUb_ = brcBuf_.Get<float>();
        if constexpr (D64_P_BROADCAST) {
            // P uses one scalar scale. Keep its repeated source block away from the score/P alignment without
            // allocating another buffer.
            quantScaleUb_ = quantScaleBuf_.Get<half>()[FP16_BLOCK_LANES];
        } else {
            quantScaleUb_ = quantScaleBuf_.Get<half>();
        }
        rowMaxUb_ = rowMaxBuf_.Get<half>();
        rowSumUb_ = rowSumBuf_.Get<half>();
        rowBrcUb_ = rowBrcBuf_.Get<half>();
        rowMaxFloatUb_ = rowMaxFloatBuf_.Get<float>();
        rowSumFloatUb_ = rowSumFloatBuf_.Get<float>();
        Duplicate(quantScaleUb_, static_cast<half>(pScaleValue_),
                  D64_P_BROADCAST ? FP16_BLOCK_LANES : FP16_REPEAT_LANES);
        PipeBarrier<PIPE_V>();

        if constexpr (D96_PREFILL && PATH == Int8GqaPath::BSND) {
            if (tiling_.groupSize == 1U && tiling_.maskMode == Int8GqaTnd::QFA_MASK_CAUSAL_TILING) {
                // BSND's row sum only uses the first 32 FP32 entries. Its existing dead tail holds this immutable
                // LUT, including N tails.
                static_assert(MAX_SEGMENTS * MAX_REDUCE_ROWS * sizeof(float) >=
                              CAUSAL_MASK_LUT_OFFSET_BYTES + (CAUSAL_MASK_WORD_BITS + 1U) * sizeof(uint32_t));
                auto maskLut =
                    rowSumFloatBuf_.Get<uint8_t>()[CAUSAL_MASK_LUT_OFFSET_BYTES].template ReinterpretCast<uint32_t>();
                for (uint32_t bits = 0U; bits < CAUSAL_MASK_WORD_BITS; ++bits)
                    maskLut.SetValue(bits, (1U << bits) - 1U);
                maskLut.SetValue(CAUSAL_MASK_WORD_BITS, UINT32_MAX);
                QfaSyncEvent<HardEvent::S_V>(EVENT_ID6);
            }
        }
    }

    __aicore__ inline bool CachedKScales(uint32_t nStart = 0U) const
    {
        return Int8GqaTnd::QfaKScaleCachedTile(cacheK_, K_SCALE_PREFIX_CACHE, nStart, tiling_.blockN);
    }

    __aicore__ inline bool NeedKScale(uint32_t nStart) const
    {
        return !CachedKScales(nStart) || (kCacheValid_ & (1U << (nStart / tiling_.blockN))) == 0U;
    }

    // A prior task's Vec1/Vec2 may rebind coordinates after K production. Keep cache ownership tied to the current
    // producer instead of BeginTask.
    __aicore__ inline void SelectKTask(uint32_t batch, uint32_t kvHead)
    {
        // K production only needs the sequence and KV head. Leave live softmax row ownership to the subsequent
        // Vec1/Vec2 stages.
        batch_ = batch;
        kvHead_ = kvHead;
        if (cacheK_ && (lastKBatch_ != batch || lastKHead_ != kvHead)) {
            kCacheValid_ = 0U;
            lastKBatch_ = batch;
            lastKHead_ = kvHead;
        }
    }

    __aicore__ inline void SetSequence(uint32_t sq, uint32_t sk, uint32_t qStart, uint32_t kStart)
    {
        tiling_.seqLenQ = sq;
        tiling_.seqLenKv = sk;
        tiling_.qTokenStart = qStart;
        tiling_.kvTokenStart = kStart;
    }

    // Rebind a staged task without loading Q descale or resetting online state.
    __aicore__ inline void SelectTask(uint32_t batch, uint32_t kvHead, uint32_t mStart, uint32_t mSize,
                                      bool selectVScale = true)
    {
        batch_ = batch;
        kvHead_ = kvHead;
        if (selectVScale && lastVHead_ != kvHead) {
            pvFixInv_ = 1.0f / PvFixpipeCompensation(vDescaleGm_.GetValue(kvHead), pScaleValue_, tiling_.blockN);
            lastVHead_ = kvHead;
        }
        mStart_ = mStart;
        uint32_t halfRows = BlockM() / Int8GqaTnd::QFA_VECTOR_CORES_PER_CUBE;

        rowStart_ = subBlock_ * halfRows;
        rows_ = (mSize > rowStart_) ? Min32(halfRows, mSize - rowStart_) : 0U;
    }

    __aicore__ inline void BeginTask(uint32_t batch, uint32_t kvHead, uint32_t mStart, uint32_t mSize,
                                     bool selectVScale = true, bool selectKScale = true)
    {
        if (selectKScale)
            SelectKTask(batch, kvHead);
        SelectTask(batch, kvHead, mStart, mSize, selectVScale);
        stateIdx_ = 0U;
        haveAccum_ = false;
        if (rows_ == 0U) {
            return;
        }

        const uint32_t flatStart = mStart + rowStart_;
        const uint32_t firstToken = flatStart / tiling_.groupSize;
        const uint32_t tokenCount = (flatStart + rows_ + tiling_.groupSize - 1U) / tiling_.groupSize - firstToken;
        const uint64_t rawQCount = static_cast<uint64_t>(tokenCount) * tiling_.numHeadsQ;
        const bool contiguousQ = PATH != Int8GqaPath::BNSD && QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) &&
                                 tiling_.numHeadsKv == 1U;
        const bool packedQ = PATH != Int8GqaPath::BNSD && QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) &&
                             !contiguousQ && tiling_.groupSize > 1U && tiling_.groupSize < FP32_BLOCK_LANES;
        const bool bulkQ = PATH != Int8GqaPath::BNSD && QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) &&
                           !contiguousQ && !packedQ &&
                           HeadDimAlign() <= Int8GqaTnd::QFA_LARGE_FP32_HEAD_DIM_THRESHOLD &&
                           rawQCount <= MAX_BULK_Q_SCALE_ELEMENTS;
        const bool alignedQ = PATH != Int8GqaPath::BNSD && QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) &&
                              !contiguousQ && !bulkQ && tiling_.groupSize % FP32_BLOCK_LANES == 0U;
        // Cache one Q-scale tile for all KV iterations.
        const uint64_t qOffset =
            (static_cast<uint64_t>(batch) * tiling_.numHeadsQ + static_cast<uint64_t>(kvHead) * tiling_.groupSize) *
                tiling_.seqLenQ +
            mStart + rowStart_;
        QfaSyncEvent<HardEvent::V_MTE2>(EVENT_ID6);
        DataCopyParams qCopy(1U, rows_ * sizeof(float), 0U, 0U);
        DataCopyPadParams qPad{true, 0U, static_cast<uint8_t>(Align8(rows_) - rows_), 0U};
        if (tiling_.layoutQ == Int8GqaTnd::QFA_LAYOUT_BNSD) {
            DataCopyPad(tmpBuf_.Get<float>(), qDescaleGm_[qOffset], qCopy, qPad);
        } else if (contiguousQ) {
            const uint64_t offset =
                QfaRow(tiling_.layoutQ, batch, 0U, 0U, tiling_.numHeadsQ, tiling_.seqLenQ, tiling_.qTokenStart) +
                flatStart;
            DataCopyPad(tmpBuf_.Get<float>(), qDescaleGm_[offset], qCopy, qPad);
        } else if (packedQ) {
            const uint64_t offset = QfaRow(tiling_.layoutQ, batch, kvHead * tiling_.groupSize, firstToken,
                                           tiling_.numHeadsQ, tiling_.seqLenQ, tiling_.qTokenStart);
            DataCopyExtParams copy{
                static_cast<uint16_t>(tokenCount), tiling_.groupSize * static_cast<uint32_t>(sizeof(float)),
                (tiling_.numHeadsQ - tiling_.groupSize) * static_cast<uint32_t>(sizeof(float)), 0U, 0U};
            DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>(FP32_BLOCK_LANES - tiling_.groupSize), 0.0f};
            DataCopyPad(tmpBuf_.Get<uint8_t>()[STRIDED_SCALE_TMP_OFFSET_BYTES].ReinterpretCast<float>(),
                        qDescaleGm_[offset], copy, pad);
        } else if (bulkQ) {
            const uint64_t offset =
                QfaRow(tiling_.layoutQ, batch, 0U, firstToken, tiling_.numHeadsQ, tiling_.seqLenQ, tiling_.qTokenStart);
            const uint32_t count = static_cast<uint32_t>(rawQCount);
            DataCopyExtParams copy{1U, count * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>(Align8(count) - count), 0.0f};
            DataCopyPad(tmpBuf_.Get<uint8_t>()[STRIDED_SCALE_TMP_OFFSET_BYTES].ReinterpretCast<float>(),
                        qDescaleGm_[offset], copy, pad);
        } else {
            LocalTensor<float> strided =
                tmpBuf_.Get<uint8_t>()[STRIDED_SCALE_TMP_OFFSET_BYTES].ReinterpretCast<float>();
            for (uint32_t done = 0; done < rows_;) {
                const uint32_t row = flatStart + done;
                const uint32_t token = QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, row);
                const uint32_t h = QuantFlashAttnLayoutInt8<PATH>::QfaQueryHead(tiling_, kvHead, row);
                const uint32_t count = QuantFlashAttnLayoutInt8<PATH>::QfaQueryRun(tiling_, row, rows_ - done);
                const uint64_t offset =
                    QfaRow(tiling_.layoutQ, batch, h, token, tiling_.numHeadsQ, tiling_.seqLenQ, tiling_.qTokenStart);
                if (alignedQ) {
                    DataCopyExtParams copy{1U, count * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                    DataCopyPad(tmpBuf_.Get<float>()[done], qDescaleGm_[offset], copy,
                                DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
                } else {
                    DataCopyExtParams copy{static_cast<uint16_t>(count), sizeof(float),
                                           (QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) ?
                                                0U :
                                                (tiling_.numHeadsQ - 1U) * static_cast<uint32_t>(sizeof(float))),
                                           0U, 0U};
                    DataCopyPad(strided[done * FP32_BLOCK_LANES], qDescaleGm_[offset], copy,
                                DataCopyPadExtParams<float>{true, 0U, FP32_BLOCK_LANES - 1U, 0.0f});
                }
                done += count;
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
        // The first softmax/PV tile overwrites row state and accumulation.
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        if (packedQ) {
            CompactSgGroups(flatStart % tiling_.groupSize, rows_, tokenCount);
        } else if (bulkQ) {
            GatherSgScales(flatStart % tiling_.groupSize, rows_);
        } else if (QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) && !contiguousQ && !alignedQ) {
            CompactStridedScales(tmpBuf_.Get<float>(), rows_);
        }
        Cast(qScaleBuf_.Get<half>(), tmpBuf_.Get<float>(), RoundMode::CAST_NONE, Align8(rows_));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareKScale(uint32_t nStart, uint32_t nSize, uint32_t slot)
    {
        if constexpr (PATH == Int8GqaPath::BNSD) {
            SelectKTask(batch_, kvHead_);
            if (!NeedKScale(nStart))
                return;
            if (cacheK_) {
                for (uint32_t start = 0U; start < tiling_.seqLenKv; start += tiling_.blockN) {
                    EncodeKScaleBlock(start, Min32(tiling_.blockN, tiling_.seqLenKv - start), slot);
                }
                // Mark all tiles, including the second AIV's empty KV tail, before the caller publishes the single
                // paired READY.
                kCacheValid_ = (1U << ((tiling_.seqLenKv + tiling_.blockN - 1U) / tiling_.blockN)) - 1U;
                return;
            }
        }
        EncodeKScaleBlock(nStart, nSize, slot);
    }

    __aicore__ inline void EncodeKScaleBlock(uint32_t nStart, uint32_t nSize, uint32_t slot)
    {
        if constexpr (PATH != Int8GqaPath::BNSD) {
            if (CachedKScales(nStart))
                kCacheValid_ |= 1U << (nStart / tiling_.blockN);
        }

        const uint32_t halfStart = subBlock_ * (tiling_.blockN / Int8GqaTnd::QFA_VECTOR_CORES_PER_CUBE);
        if (nSize <= halfStart) {
            return;
        }
        const uint32_t count = Min32(tiling_.blockN / Int8GqaTnd::QFA_VECTOR_CORES_PER_CUBE, nSize - halfStart);
        // Gather processes 32 FP32 scales per repeat. Pad short KV tails to one complete repeat so their encoded
        // lanes are initialized.
        const uint32_t aligned = (count + K_SCALE_FIXPIPE_ALIGNMENT - 1U) & ~(K_SCALE_FIXPIPE_ALIGNMENT - 1U);
        LocalTensor<uint32_t> scale =
            tiling_.layoutKv == Int8GqaTnd::QFA_LAYOUT_BNSD ?
                tmpBuf_.Get<uint32_t>() :
                tmpBuf_.Get<uint8_t>()[STRIDED_SCALE_TMP_OFFSET_BYTES].ReinterpretCast<uint32_t>();

        const bool packedHeads = PATH != Int8GqaPath::BNSD && QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutKv) &&
                                 tiling_.numHeadsKv <= MAX_PACKED_KV_HEADS &&
                                 static_cast<uint64_t>(aligned) * tiling_.numHeadsKv * sizeof(float) <=
                                     TmpBytes() - STRIDED_SCALE_TMP_OFFSET_BYTES;
        const uint32_t scaleStride = tiling_.layoutKv == Int8GqaTnd::QFA_LAYOUT_BNSD ?
                                         1U :
                                         (packedHeads ? tiling_.numHeadsKv : FP32_BLOCK_LANES);
        LocalTensor<uint32_t> encoded =
            tmpBuf_.Get<uint8_t>()[K_SCALE_ENCODED_OFFSET_BYTES].ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> index = kScaleIndexBuf_.Get<uint32_t>();
        if (aligned != count)
            Duplicate(scale, 0U, aligned * scaleStride);
        // Retire Vec1/Vec2 readers before reusing their scale scratch.
        QfaSyncEvent<HardEvent::V_MTE2>(EVENT_ID6);
        const uint64_t offset = QfaRow(tiling_.layoutKv, batch_, packedHeads ? 0U : kvHead_, nStart + halfStart,
                                       tiling_.numHeadsKv, tiling_.seqLenKv, tiling_.kvTokenStart);
        if (tiling_.layoutKv == Int8GqaTnd::QFA_LAYOUT_BNSD || packedHeads) {
            const uint32_t elements = count * scaleStride;
            DataCopyExtParams copy{1U, elements * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(scale.ReinterpretCast<float>(), kDescaleGm_[offset], copy,
                        DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>(Align8(elements) - elements), 0.0f});
        } else {
            DataCopyExtParams copy{static_cast<uint16_t>(count), sizeof(float),
                                   (tiling_.numHeadsKv - 1U) * static_cast<uint32_t>(sizeof(float)), 0U, 0U};
            DataCopyPad(scale.ReinterpretCast<float>(), kDescaleGm_[offset], copy,
                        DataCopyPadExtParams<float>{true, 0U, FP32_BLOCK_LANES - 1U, 0.0f});
        }
        QfaSyncEvent<HardEvent::MTE2_V>(EVENT_ID6);
        Muls(scale.ReinterpretCast<float>(), scale.ReinterpretCast<float>(), tiling_.softmaxScale,
             aligned * scaleStride);
        if (tiling_.layoutKv != Int8GqaTnd::QFA_LAYOUT_BNSD) {
            LocalTensor<uint32_t> stridedIndex =
                tmpBuf_.Get<uint8_t>()[K_SCALE_STRIDED_INDEX_OFFSET_BYTES].ReinterpretCast<uint32_t>();
            Muls(stridedIndex.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(scaleStride), aligned * FIXPIPE_ENCODED_WORDS_PER_SCALE);
            index = stridedIndex;
        }
        Duplicate(encoded, FIXPIPE_DEQSCALE_FP16_ENABLED, FIXPIPE_ENCODED_WORDS_PER_SCALE * aligned);
        PipeBarrier<PIPE_V>();
        uint64_t mask[VECTOR_MASK_WORDS] = {FIXPIPE_SCALE_WORD_MASK, 0ULL};
        Gather(encoded, scale, index, packedHeads ? kvHead_ * sizeof(float) : 0U, mask,
               static_cast<uint8_t>(aligned / K_SCALE_FIXPIPE_ALIGNMENT), VECTOR_REPEAT_BLOCKS);
        // Round Fixpipe's FP32 bits [31:13] in the even lanes, preserving odd control words.
        PipeBarrier<PIPE_V>();
        auto bits = encoded.template ReinterpretCast<int32_t>();
        Adds<int32_t, false>(bits, bits, FIXPIPE_ROUNDING_BIAS, static_cast<uint64_t>(0),
                             static_cast<uint8_t>(aligned / K_SCALE_FIXPIPE_ALIGNMENT),
                             {1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS});
        QfaSyncEvent<HardEvent::V_MTE3>(EVENT_ID6);
        DataCopy(kDeqWorkspaceGm_[Int8GqaTnd::QfaKScaleGmOffset(cacheK_, K_SCALE_PREFIX_CACHE, nStart, slot,
                                                                tiling_.blockN) +
                                  halfStart],
                 encoded.ReinterpretCast<uint64_t>(), (count + FP16_BLOCK_LANES - 1U) & ~(FP16_BLOCK_LANES - 1U));
        QfaSyncEvent<HardEvent::MTE3_V>(EVENT_ID6);
    }

    __aicore__ inline void Vec1(uint32_t nStart, uint32_t nSize, uint32_t slot, bool isFirst)
    {
        nStart_ = nStart;
        // Preserve the logical KV block and its online P quantization domain. UB reduction only needs the valid
        // columns rounded to one vector.
        vectorColumns_ =
            Min32(tiling_.blockN, (nSize + FP16_REPEAT_LANES - 1U) / FP16_REPEAT_LANES * FP16_REPEAT_LANES);

        if (rows_ == 0U) {
            return;
        }
        const uint32_t nextIdx = stateIdx_ ^ 1U;
        // Ping-pong score/P buffers overlap DMA with softmax; D96 publishes P before reduction completes.
        const uint32_t chunkCount = (rows_ + VecRows() - 1U) / VecRows();
        constexpr uint32_t pBufferCount = P_BUFFER_COUNT;

        CopyScoreChunk(slot, 0U);
        for (uint32_t chunkIdx = 0U; chunkIdx < chunkCount; ++chunkIdx) {
            const uint32_t bufId = chunkIdx & 1U;
            if constexpr (D64_SCORE_PREFETCH) {
                // Match SplitFuse FIA's preLoad=1 row loop: issue the next independent score DMA before waiting for
                // the current one.
                if (chunkIdx + 1U < chunkCount) {
                    CopyScoreChunk(slot, chunkIdx + 1U);
                }
            }
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1 + bufId);
            if constexpr (!(D64_SCORE_PREFETCH)) {
                if (chunkIdx + 1U < chunkCount) {
                    CopyScoreChunk(slot, chunkIdx + 1U);
                }
            }
            const uint32_t chunk = chunkIdx * VecRows();
            const uint32_t chunkRows = Min32(VecRows(), rows_ - chunk);
            const uint32_t chunkRowsAligned = Align8(chunkRows);
            {
                const uint32_t columns = VectorColumns();
                scoreUb_ = dataBuf_.Get<half>()[bufId * VecRows() * MAX_BLOCK_N];
                if (chunkIdx >= pBufferCount) {
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1 + bufId);
                }
                pUb_ = tmpBuf_.Get<int8_t>()[bufId * VecRows() * MAX_BLOCK_N];

                // MM1 Fixpipe has already fused softmaxScale into K descale.
                const half negInf = static_cast<half>(-__builtin_inff());
                MaskScoreTail(nSize, chunkRows);
                if (NeedsCausalMask(chunk, chunkRows, nSize)) {
                    ApplyCausalMask(chunkRows, chunk, nSize, static_cast<half>(-QFA_FP16_MAX_FINITE));
                }
                for (uint32_t row = chunkRows; row < chunkRowsAligned; ++row) {
                    // A fully -inf row would produce -inf - (-inf) after its FP32 max is cast to FP16. Inactive
                    // rows are discarded, so a finite fill avoids NaNs without affecting valid rows.
                    Duplicate(scoreUb_[row * columns], static_cast<half>(-QFA_FP16_MAX_FINITE), columns);
                }
                PipeBarrier<PIPE_V>();

                OnlineSoftmaxHalf(chunkRowsAligned, slot, chunk, nextIdx, nSize, isFirst);
                if constexpr (!D96_PREFILL)
                    PublishP(slot, chunk, chunkRows);
            }
        }
        for (uint32_t i = 0; i < Min32(chunkCount, pBufferCount); ++i) {
            const uint32_t pBufId = (chunkCount - 1U - i) & 1U;
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1 + pBufId);
        }
        stateIdx_ = nextIdx;
    }

    __aicore__ inline void SetSoftmaxState(uint32_t idx)
    {
        stateIdx_ = idx;
    }

    __aicore__ inline void Vec2Task(uint32_t slot, bool isFirst, bool isLast, uint32_t bank)
    {
        haveAccum_ = !isFirst;
        const uint32_t savedStateIdx = stateIdx_;
        if (isLast)
            stateIdx_ = bank;
        Vec2(slot, isLast);
        stateIdx_ = savedStateIdx;
    }

    __aicore__ inline void Vec2(uint32_t slot, bool isLast)
    {
        if (rows_ == 0U)
            return;
        if constexpr (RETURN_LSE) {
            if (isLast)
                WriteLse(false);
        }
        // Keep each pipeline's validated accumulation precision and row policy.
        if (D96_PREFILL ||
            (BlockM() == Int8GqaTnd::QFA_LARGE_QUERY_TILE_ROWS &&
             HeadDimAlign() <= Int8GqaTnd::QFA_LARGE_FP32_HEAD_DIM_THRESHOLD) ||
            partitionedMhaHalf_ ||
            (BlockM() == Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE &&
             HeadDimAlign() > Int8GqaTnd::QFA_LARGE_FP32_HEAD_DIM_THRESHOLD && tiling_.seqLenKv <= tiling_.blockN))
            AccumulateOutput<half>(slot, isLast);
        else
            AccumulateOutput<float>(slot, isLast);
        haveAccum_ = true;
    }

    __aicore__ inline void WriteZero()
    {
        if constexpr (RETURN_LSE)
            WriteLse(true);
        for (uint32_t chunk = 0; chunk < rows_; chunk += VecRows()) {
            const uint32_t count = Min32(VecRows(), rows_ - chunk);
            Duplicate(pvUb_, static_cast<half>(0.0f), count * HeadDimAlign());
            PipeBarrier<PIPE_V>();
            CopyOut(chunk, count);
        }
    }

private:
    // Called exactly once per completed Q task, before any PV compensation changes its denominator. The two online
    // max/sum banks have the same lifetime, including the interleaved cross-Q pipeline.
    __aicore__ inline void WriteLse(bool empty)
    {
        if constexpr (RETURN_LSE) {
            if (rows_ == 0U)
                return;
            // Vec1 has published P and retired its score DMA; Vec2 has not loaded PV yet. Reuse 3 KiB of that dead
            // score/PV buffer, leaving the broadcast and online-state buffers untouched.
            LocalTensor<float> values = dataBuf_.Get<float>();
            if (empty) {
                Duplicate(values, -__builtin_inff(), Align8(rows_));
            } else {
                Ln(values, sumStateUb_[stateIdx_ * MAX_TASK_ROWS], Align8(rows_));
                PipeBarrier<PIPE_V>();
                Add(values, values, maxStateUb_[stateIdx_ * MAX_TASK_ROWS], Align8(rows_));
            }
            PipeBarrier<PIPE_V>();
            if (!empty && tiling_.maskMode != 0U && tiling_.seqLenQ > tiling_.seqLenKv) {
                QfaSyncEvent<HardEvent::V_S>(EVENT_ID6);
                for (uint32_t r = 0U; r < rows_; ++r) {
                    const uint32_t token =
                        QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, mStart_ + rowStart_ + r);
                    if (token < tiling_.seqLenQ - tiling_.seqLenKv)
                        values.SetValue(r, -__builtin_inff());
                }
                QfaSyncEvent<HardEvent::S_V>(EVENT_ID6);
            }
            const uint32_t first = mStart_ + rowStart_;
            const bool tokenMajor = QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ);
            const uint32_t stride = tokenMajor ? tiling_.groupSize : 1U;
            const uint32_t runs = Min32(stride, rows_);
            auto packed = values[MAX_TASK_ROWS];
            // Keep the raw and gathered LSE rows ahead of the gather indices.
            LocalTensor<int32_t> indices = dataBuf_.Get<int32_t>()[2U * MAX_TASK_ROWS];
            if (stride > 1U) {
                // Immutable row indices avoid scalar stores into retiring vector scratch.
                const auto linear = qScaleBuf_.Get<uint8_t>()[MAX_TASK_ROWS * sizeof(half)].ReinterpretCast<int32_t>();
                Muls(indices, linear, static_cast<int32_t>(stride * sizeof(float)), Align8(rows_));
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t j = 0U; j < runs; ++j) {
                const uint32_t count = (rows_ - 1U - j) / stride + 1U;
                const uint32_t row = first + j;
                const uint32_t head = QuantFlashAttnLayoutInt8<PATH>::QfaQueryHead(tiling_, kvHead_, row);
                const uint32_t token = QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, row);
                const uint64_t offset =
                    PATH == Int8GqaPath::TND ?
                        static_cast<uint64_t>(head) * tiling_.totalTokensQ + tiling_.qTokenStart + token :
                        (static_cast<uint64_t>(batch_) * tiling_.numHeadsQ + head) * tiling_.seqLenQ + token;
                // Head-major Q rows can cross a head boundary contiguously.
                auto source = values;
                if (stride > 1U) {
                    Gather(packed, values, indices.ReinterpretCast<uint32_t>(), j * sizeof(float), count);
                    PipeBarrier<PIPE_V>();
                    source = packed;
                }
                QfaSyncEvent<HardEvent::V_MTE3>(EVENT_ID6);
                DataCopyExtParams copy{};
                copy.blockCount = 1U;
                copy.blockLen = count * sizeof(float);
                DataCopyPad(this->lseGm_[offset], source, copy);
                QfaSyncEvent<HardEvent::MTE3_V>(EVENT_ID6);
            }
            ResetMask();
            QfaSyncEvent<HardEvent::MTE3_S>(EVENT_ID6);
        }
    }

    static_assert((D64_VEC_FEATURES & ~Int8GqaTnd::QFA_VEC_PREFILL_FEATURES) == 0U, "Unsupported vector policy");
    static constexpr bool D64_P_BROADCAST = (D64_VEC_FEATURES & Int8GqaTnd::QFA_VEC_P_BROADCAST) != 0U;
    static constexpr bool D64_SCORE_PREFETCH = (D64_VEC_FEATURES & Int8GqaTnd::QFA_VEC_SCORE_PREFETCH) != 0U;
    static constexpr bool D64_DIRECT_FIRST_PV = (D64_VEC_FEATURES & Int8GqaTnd::QFA_VEC_DIRECT_FIRST_PV) != 0U;
    static constexpr bool D64_OVERLAP_COEFF = (D64_VEC_FEATURES & Int8GqaTnd::QFA_VEC_OVERLAP_COEFF) != 0U;
    static constexpr uint32_t SLOT_COUNT = Int8GqaTnd::QFA_PIPELINE_SLOTS;
    static constexpr uint32_t P_BUFFER_COUNT = 2U;
    static constexpr uint32_t ONLINE_STATE_BUFFER_COUNT = 2U; // Alternate previous and current max/sum.
    static constexpr uint32_t FP16_BLOCK_LANES = QFA_INT8_DMA_BLOCK_BYTES / sizeof(half);
    static constexpr uint32_t FP32_BLOCK_LANES = QFA_INT8_DMA_BLOCK_BYTES / sizeof(float);
    static constexpr uint32_t VECTOR_REPEAT_BYTES = 256U;
    static constexpr uint32_t VECTOR_REPEAT_BLOCKS = VECTOR_REPEAT_BYTES / QFA_INT8_DMA_BLOCK_BYTES;
    static constexpr uint32_t FP16_REPEAT_LANES = VECTOR_REPEAT_BYTES / sizeof(half);
    static constexpr uint32_t MAX_VECTOR_REPEATS = 255U;
    static constexpr uint32_t VECTOR_MASK_WORDS = 2U;          // AscendC vector bitmasks use two uint64_t words.
    static constexpr uint32_t K_SCALE_FIXPIPE_ALIGNMENT = 32U; // Pack and round scales in whole vector repeats.
    static constexpr uint32_t FP32_BYTE_OFFSET_SHIFT = 2U;     // log2(sizeof(float)).
    static constexpr uint32_t FIXPIPE_ENCODED_WORDS_PER_SCALE = sizeof(uint64_t) / sizeof(uint32_t);
    static constexpr uint32_t PV_OVERLAY_ROWS = 128U; // Cube writes FP16 PV into 128-row P regions.
    static constexpr uint32_t LARGE_FP32_VEC_ROWS = 16U;
    static constexpr uint32_t MAX_BROADCAST_ROWS = 64U;
    static constexpr uint32_t MAX_PACKED_KV_HEADS = 16U;
    static constexpr float MIN_FP32_SOFTMAX_DENOMINATOR = 1.0e-20f;
    static constexpr uint32_t STRIDED_SCALE_TMP_OFFSET_BYTES = 8192U;
    static constexpr uint32_t P_MASK_ROW_OFFSET_BYTES = 4096U;
    static constexpr uint32_t P_MASK_BITMAP_OFFSET_BYTES = 8192U;
    static constexpr uint32_t P_MASK_FILL_OFFSET_BYTES = 12288U;
    static constexpr uint32_t CAUSAL_MASK_WORD_BITS = 32U;
    static constexpr uint32_t CAUSAL_MASK_LUT_OFFSET_BYTES = 256U;
    static constexpr uint32_t SG_SCALE_MASK_BYTE_BROADCAST = 0x01010101U;
    static constexpr uint32_t FIXPIPE_DEQSCALE_FP16_ENABLED = 0x4000U;
    static constexpr int32_t FIXPIPE_ROUNDING_BIAS = 0x1000;
    static constexpr uint64_t FIXPIPE_SCALE_WORD_MASK = 0x5555555555555555ULL;
    static constexpr uint32_t DEFAULT_VEC1_ROWS = 32U;
    static constexpr uint32_t WIDE_VEC1_ROWS = 48U;
    static constexpr uint32_t STANDARD_VEC_ROWS =
        D96_PREFILL ? (STATIC_M == Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE ? WIDE_VEC1_ROWS : DEFAULT_VEC1_ROWS) :
        STATIC_D != 0U && STATIC_D <= Int8GqaTnd::QFA_HEAD_DIM_64 ? WIDE_VEC1_ROWS :
                                                                    DEFAULT_VEC1_ROWS;
    static constexpr uint32_t MAX_REDUCE_ROWS = (STANDARD_VEC_ROWS + FP16_BLOCK_LANES - 1U) & ~(FP16_BLOCK_LANES - 1U);
    static constexpr uint32_t MAX_BLOCK_N = Int8GqaTnd::QFA_DEFAULT_KV_TILE_TOKENS;

    static constexpr uint32_t MAX_TASK_ROWS = 256U;
    static constexpr uint32_t MAX_BULK_Q_SCALE_ELEMENTS = 6144U;
    static constexpr uint32_t K_SCALE_ENCODED_OFFSET_BYTES = 2048U;
    static constexpr uint32_t K_SCALE_STRIDED_INDEX_OFFSET_BYTES = 4096U;
    static constexpr uint32_t SG_SCALE_BITMAP_OFFSET_BYTES = 4096U;
    static constexpr uint32_t SG_SCALE_GATHER_INDEX_OFFSET_BYTES = 4096U;
    static constexpr uint32_t SG_SCALE_GATHER_NUMERATOR_OFFSET_BYTES =
        SG_SCALE_GATHER_INDEX_OFFSET_BYTES + MAX_TASK_ROWS * sizeof(int32_t);
    static constexpr uint32_t SG_SCALE_GATHER_DENOMINATOR_OFFSET_BYTES =
        SG_SCALE_GATHER_NUMERATOR_OFFSET_BYTES + MAX_TASK_ROWS * sizeof(float);
    static_assert(SG_SCALE_GATHER_DENOMINATOR_OFFSET_BYTES + MAX_TASK_ROWS * sizeof(float) <=
                  STRIDED_SCALE_TMP_OFFSET_BYTES);
    static constexpr float SG_SCALE_GATHER_FLOOR_EPSILON = 0.00001f;
    static constexpr uint32_t MHA_PARTITIONED_ACCUMULATOR_QUERY_THRESHOLD = 1024U;
    static constexpr uint32_t D64_WIDE_VEC2_MIN_KV_TOKENS = 16384U;
    static constexpr uint32_t OUTPUT_CAST_CHUNK_ELEMENTS = 4096U;
    static constexpr uint32_t TAIL_MASK_LANES = FP16_BLOCK_LANES;
    static constexpr uint32_t TAIL_MASK_ALIGNMENT_BITS = TAIL_MASK_LANES - 1U;
    static constexpr uint64_t TAIL_MASK_ALL_LANES = (1ULL << TAIL_MASK_LANES) - 1ULL;
    static constexpr uint32_t MAX_ACC_ELEMS = 16384U; // 128x128 or 64x256 FP32; 256x128 FP16.
    static constexpr uint32_t MAX_SEGMENTS = MAX_BLOCK_N / FP16_REPEAT_LANES;
    __aicore__ inline uint32_t VectorColumns() const
    {
        return vectorColumns_;
    }
    __aicore__ inline uint32_t VecRows() const
    {
        return largeFp32Accumulator_ ? LARGE_FP32_VEC_ROWS : STANDARD_VEC_ROWS;
    }
    __aicore__ inline uint32_t TmpBytes() const
    {
        return P_BUFFER_COUNT * VecRows() * MAX_BLOCK_N;
    }

    __aicore__ inline uint32_t HeadDim() const
    {
        return STATIC_D != 0U ? STATIC_D : tiling_.headDim;
    }
    __aicore__ inline uint32_t OutputDim() const
    {
        return tiling_.outputHeadDim != 0U ? tiling_.outputHeadDim : HeadDim();
    }
    __aicore__ inline uint32_t HeadDimAlign() const
    {
        return STATIC_D != 0U ?
                   ((STATIC_D + Int8GqaTnd::QFA_HEAD_DIM_ALIGNMENT - 1U) & ~(Int8GqaTnd::QFA_HEAD_DIM_ALIGNMENT - 1U)) :
                   tiling_.headDimAlign;
    }
    __aicore__ inline uint32_t BlockM() const
    {
        return STATIC_M != 0U ? STATIC_M : tiling_.blockM;
    }
    __aicore__ inline static uint32_t Min32(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }
    __aicore__ inline static uint32_t Align8(uint32_t x)
    {
        return (x + FP32_BLOCK_LANES - 1U) & ~(FP32_BLOCK_LANES - 1U);
    }

    __aicore__ inline void CopyScoreChunk(uint32_t slot, uint32_t chunkIdx)
    {
        const uint32_t chunk = chunkIdx * VecRows();
        const uint32_t rows = Min32(VecRows(), rows_ - chunk);
        const uint32_t bufId = chunkIdx & 1U;
        QfaSyncEvent<HardEvent::V_MTE2>(EVENT_ID1 + bufId);
        const uint32_t source = slot * tiling_.scoreSlotElems + (rowStart_ + chunk) * tiling_.blockN;
        if (VectorColumns() == tiling_.blockN) {
            DataCopy(dataBuf_.Get<half>()[bufId * VecRows() * MAX_BLOCK_N], scoreGm_[source], rows * tiling_.blockN);
        } else {
            DataCopyParams copy(static_cast<uint16_t>(rows), VectorColumns() / FP16_BLOCK_LANES,
                                (tiling_.blockN - VectorColumns()) / FP16_BLOCK_LANES, 0U);
            DataCopy(dataBuf_.Get<half>()[bufId * VecRows() * MAX_BLOCK_N], scoreGm_[source], copy);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1 + bufId);
    }

    template <typename T>
    __aicore__ inline void AccumulateOutput(uint32_t slot, bool isLast)
    {
        constexpr bool HALF = sizeof(T) == sizeof(half);
        const uint32_t dAlign = HeadDimAlign();
        constexpr uint32_t defaultVec2Rows = 64U;
        constexpr uint32_t wideFp16Vec2Rows = 128U;
        constexpr uint32_t largeDFp32Vec2Rows = 32U;
        uint32_t vec2Rows =
            largeFp32Accumulator_ ?
                LARGE_FP32_VEC_ROWS :
                (dAlign > Int8GqaTnd::QFA_LARGE_FP32_HEAD_DIM_THRESHOLD ? largeDFp32Vec2Rows : defaultVec2Rows);
        LocalTensor<T> brc = brcBuf_.Get<T>();
        if constexpr (HALF) {
            vec2Rows =
                largeFp32Accumulator_     ? LARGE_FP32_VEC_ROWS :
                PATH == Int8GqaPath::BNSD ? (D96_PREFILL || (STATIC_D == Int8GqaTnd::QFA_HEAD_DIM_64 &&
                                                             tiling_.seqLenKv >= D64_WIDE_VEC2_MIN_KV_TOKENS) ?
                                                 wideFp16Vec2Rows :
                                                 defaultVec2Rows) :
                dAlign <= Int8GqaTnd::QFA_LARGE_FP32_HEAD_DIM_THRESHOLD &&
                        (QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) ||
                         (STATIC_D == Int8GqaTnd::QFA_HEAD_DIM_64 && tiling_.seqLenKv >= D64_WIDE_VEC2_MIN_KV_TOKENS)) ?
                                            wideFp16Vec2Rows :
                                            defaultVec2Rows;
            if (STATIC_D == Int8GqaTnd::QFA_HEAD_DIM_64 || (PATH == Int8GqaPath::BNSD && D96_PREFILL) ||
                (PATH != Int8GqaPath::BNSD && QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ)))
                brc = dataBuf_.Get<T>()[vec2Rows * dAlign];
            if constexpr (PATH == Int8GqaPath::TND && STATIC_D == Int8GqaTnd::QFA_HEAD_DIM_128 &&
                          STATIC_M == Int8GqaTnd::QFA_LARGE_QUERY_TILE_ROWS &&
                          (D64_VEC_FEATURES & Int8GqaTnd::QFA_VEC_D64_TASK_MAP_FEATURES) ==
                              Int8GqaTnd::QFA_VEC_D64_TASK_MAP_FEATURES)
                brc = brc[FP16_BLOCK_LANES];
        }
        for (uint32_t chunk = 0; chunk < rows_; chunk += vec2Rows) {
            const uint32_t chunkRows = Min32(vec2Rows, rows_ - chunk), elemCount = chunkRows * dAlign;
            const uint32_t pvRow = rowStart_ + chunk;
            // Each AIV owns half a P/PV slot, including when the other subcore advances.
            const uint32_t pvOffset = overlayPv_ ? slot * tiling_.pSlotElems / static_cast<uint32_t>(sizeof(half)) +
                                                       (pvRow / PV_OVERLAY_ROWS) * PV_OVERLAY_ROWS *
                                                           (tiling_.blockN / Int8GqaTnd::QFA_VECTOR_CORES_PER_CUBE) +
                                                       (pvRow % PV_OVERLAY_ROWS) * dAlign :
                                                   slot * tiling_.pvSlotElems + pvRow * dAlign;
            QfaSyncEvent<HardEvent::V_MTE2>(EVENT_ID0);
            auto accumulator = accBuf_.Get<T>()[chunk * dAlign];
            if constexpr (HALF && D64_DIRECT_FIRST_PV)
                DataCopy(!haveAccum_ ? accumulator : pvUb_, pvGm_[pvOffset], elemCount);
            else
                DataCopy(pvUb_, pvGm_[pvOffset], elemCount);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (!(HALF && D64_OVERLAP_COEFF))
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            LocalTensor<T> pv = dataBuf_.Get<T>();
            if constexpr (!HALF) {
                if (dAlign > HeadDim()) {
                    const uint32_t blockStart = HeadDim() & ~TAIL_MASK_ALIGNMENT_BITS;
                    const uint32_t alignedTail =
                        Min32((HeadDim() + TAIL_MASK_ALIGNMENT_BITS) & ~TAIL_MASK_ALIGNMENT_BITS, dAlign);
                    if (alignedTail > HeadDim()) {
                        uint64_t mask[VECTOR_MASK_WORDS] = {
                            TAIL_MASK_ALL_LANES & ~((1ULL << (HeadDim() - blockStart)) - 1ULL), 0ULL};
                        Duplicate(pvUb_[blockStart], static_cast<half>(0), mask, static_cast<uint8_t>(chunkRows), 1,
                                  dAlign / FP16_BLOCK_LANES);
                    }
                    for (uint32_t row = 0; row < chunkRows; ++row) {
                        if (alignedTail < dAlign)
                            Duplicate(pvUb_[row * dAlign + alignedTail], static_cast<half>(0), dAlign - alignedTail);
                    }
                    PipeBarrier<PIPE_V>();
                }
                pv = tmpBuf_.Get<T>();
                Cast(pv, pvUb_, RoundMode::CAST_NONE, elemCount);
                PipeBarrier<PIPE_V>();
            }
            const bool directOutput = !haveAccum_ && isLast && !(HALF && D64_DIRECT_FIRST_PV);
            auto acc = directOutput ? pv : accumulator;
            if (!haveAccum_) {
                if constexpr (HALF && D64_OVERLAP_COEFF)
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr (!(HALF && D64_DIRECT_FIRST_PV)) {
                    if (!directOutput)
                        Adds(acc, pv, static_cast<T>(0), elemCount);
                }
            } else {
                auto coefficient = coeffUb_[slot * MAX_TASK_ROWS + chunk];
                if constexpr (HALF) {
                    auto coeffHalf = tmpBuf_.Get<half>();
                    Cast(coeffHalf, coefficient, RoundMode::CAST_NONE, Align8(chunkRows));
                    PipeBarrier<PIPE_V>();
                    Brcb(brc, coeffHalf, Align8(chunkRows) / FP32_BLOCK_LANES, {1, VECTOR_REPEAT_BLOCKS});
                } else {
                    Brcb(brc, coefficient, Align8(chunkRows) / FP32_BLOCK_LANES, {1, VECTOR_REPEAT_BLOCKS});
                }
                PipeBarrier<PIPE_V>();
                if constexpr (HALF && D64_OVERLAP_COEFF)
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                OutputRowOperation<false>(acc, pv, brc, chunkRows);
            }
            PipeBarrier<PIPE_V>();
            if (isLast) {
                auto denominator = sumStateUb_[stateIdx_ * MAX_TASK_ROWS + chunk];
                if (pvFixInv_ != 1.0f) {
                    Muls(denominator, denominator, pvFixInv_, Align8(chunkRows));
                    PipeBarrier<PIPE_V>();
                }
                if constexpr (!HALF) {
                    if (WINDOWED || tiling_.maskMode != 0U) {
                        Maxs(denominator, denominator, MIN_FP32_SOFTMAX_DENOMINATOR, Align8(chunkRows));
                        PipeBarrier<PIPE_V>();
                    }
                    Brcb(brc, denominator, Align8(chunkRows) / FP32_BLOCK_LANES, {1, VECTOR_REPEAT_BLOCKS});
                } else {
                    auto coeffHalf = tmpBuf_.Get<half>();
                    Cast(coeffHalf, denominator, RoundMode::CAST_NONE, Align8(chunkRows));
                    PipeBarrier<PIPE_V>();
                    Brcb(brc, coeffHalf, Align8(chunkRows) / FP32_BLOCK_LANES, {1, VECTOR_REPEAT_BLOCKS});
                }
                PipeBarrier<PIPE_V>();
                OutputRowOperation<true>(acc, pv, brc, chunkRows);
                PipeBarrier<PIPE_V>();
                if constexpr (!HALF)
                    Cast(pvUb_, acc, RoundMode::CAST_NONE, elemCount);
                else if (!directOutput)
                    Adds(pvUb_, acc, static_cast<half>(0), elemCount);
                PipeBarrier<PIPE_V>();
                CopyOut(chunk, chunkRows);
            }
        }
    }

    __aicore__ inline void PublishP(uint32_t slot, uint32_t chunk, uint32_t validRows)
    {
        // Publish P as soon as MulCast finishes. Its selected UB slot is read-only during the destructive score sum
        // below; the next reuse waits for this MTE3 before writing that P slot again.
        const uint32_t columns = VectorColumns();
        const uint32_t pOffset = slot * tiling_.pSlotElems + (rowStart_ + chunk) * tiling_.blockN;
        const uint32_t pBufId = (chunk / VecRows()) & 1U;
        QfaSyncEvent<HardEvent::V_MTE3>(EVENT_ID1 + pBufId);
        if (columns == tiling_.blockN) {
            DataCopy(pGm_[pOffset], pUb_, validRows * columns);
        } else {
            DataCopyParams copy(static_cast<uint16_t>(validRows), columns / QFA_INT8_DMA_BLOCK_BYTES, 0U,
                                (tiling_.blockN - columns) / QFA_INT8_DMA_BLOCK_BYTES);
            DataCopy(pGm_[pOffset], pUb_, copy);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1 + pBufId);
    }

    __aicore__ inline void MaskScoreTail(uint32_t nSize, uint32_t rows)
    {
        if (nSize < VectorColumns()) {
            MaskSpan(0U, rows, nSize, VectorColumns(), static_cast<half>(-__builtin_inff()));
            PipeBarrier<PIPE_V>();
        }
    }

    // Materialize Q dequantization in FP16 before max subtraction so the rounding feeding static INT8 P follows the
    // FP16 reference steps.
    __aicore__ inline void OnlineSoftmaxHalf(uint32_t rowCount, uint32_t slot, uint32_t chunk, uint32_t nextIdx,
                                             uint32_t nSize, bool isFirst)
    {
        const uint32_t segments = VectorColumns() / FP16_REPEAT_LANES;
        const uint32_t rowStride = VectorColumns() / FP16_BLOCK_LANES; // 32-byte blocks
        LocalTensor<float> oldMax = maxStateUb_[stateIdx_ * MAX_TASK_ROWS + chunk];
        LocalTensor<float> newMax = maxStateUb_[nextIdx * MAX_TASK_ROWS + chunk];
        LocalTensor<float> oldSum = sumStateUb_[stateIdx_ * MAX_TASK_ROWS + chunk];
        LocalTensor<float> newSum = sumStateUb_[nextIdx * MAX_TASK_ROWS + chunk];
        LocalTensor<float> slotCoeff = coeffUb_[slot * MAX_TASK_ROWS + chunk];

        // Match FIA's pairwise row maximum across four 128-column segments. The fallback handles other tile widths
        // without changing the score.
        if (segments == MAX_SEGMENTS && tiling_.seqLenKv > MAX_BLOCK_N) {
            // The current P buffer is free after its MTE3 completion wait. Its full extent is reused here before
            // MulCast writes P8.
            LocalTensor<half> maxTile0 = pUb_.template ReinterpretCast<half>();
            LocalTensor<half> maxTile2 = maxTile0[VecRows() * FP16_REPEAT_LANES];
            BinaryRepeatParams maxParams(1, 1, 1, VECTOR_REPEAT_BLOCKS, rowStride, rowStride);
            if constexpr (D64_REUSE_MASK)
                SetVectorMask<half>(FP16_REPEAT_LANES);
            Max<half, !D64_REUSE_MASK>(maxTile0, scoreUb_, scoreUb_[FP16_REPEAT_LANES], FP16_REPEAT_LANES,
                                       static_cast<uint8_t>(rowCount), maxParams);
            Max<half, !D64_REUSE_MASK>(maxTile2, scoreUb_[2U * FP16_REPEAT_LANES], scoreUb_[3U * FP16_REPEAT_LANES],
                                       FP16_REPEAT_LANES, static_cast<uint8_t>(rowCount), maxParams);
            PipeBarrier<PIPE_V>();
            maxParams.src0RepStride = VECTOR_REPEAT_BLOCKS;
            maxParams.src1RepStride = VECTOR_REPEAT_BLOCKS;
            Max<half, !D64_REUSE_MASK>(maxTile0, maxTile0, maxTile2, FP16_REPEAT_LANES, static_cast<uint8_t>(rowCount),
                                       maxParams);
            PipeBarrier<PIPE_V>();
            WholeReduceMax<half, !D64_REUSE_MASK>(rowMaxUb_, maxTile0, FP16_REPEAT_LANES, rowCount, 1, 1,
                                                  VECTOR_REPEAT_BLOCKS, ReduceOrder::ORDER_ONLY_VALUE);
        } else {
            for (uint32_t seg = 0; seg < segments; ++seg) {
                WholeReduceMax(rowMaxUb_[seg * MAX_REDUCE_ROWS], scoreUb_[seg * FP16_REPEAT_LANES], FP16_REPEAT_LANES,
                               rowCount, 1, 1, rowStride, ReduceOrder::ORDER_ONLY_VALUE);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t seg = 1; seg < segments; ++seg) {
                Max(rowMaxUb_, rowMaxUb_, rowMaxUb_[seg * MAX_REDUCE_ROWS], rowCount);
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_V>();
        // Positive Q descale preserves max ordering; scale the reduction before matrix subtraction.
        Mul(rowMaxUb_, rowMaxUb_, qScaleBuf_.Get<half>()[chunk], rowCount);
        PipeBarrier<PIPE_V>();
        // Repair fully masked rows before converting their maxima.
        if (NeedsCausalMask(chunk, rowCount, 1U)) {
            QfaSyncEvent<HardEvent::V_S>(EVENT_ID6);
            for (uint32_t row = 0; row < rowCount; ++row) {
                if (CausalColumns(chunk + row, nSize) == 0U) {
                    rowMaxUb_.SetValue(row, static_cast<half>(-QFA_FP16_MAX_FINITE));
                }
            }
            QfaSyncEvent<HardEvent::S_V>(EVENT_ID6);
        }
        Cast(rowMaxFloatUb_, rowMaxUb_, RoundMode::CAST_NONE, rowCount);
        PipeBarrier<PIPE_V>();
        if (isFirst) {
            // Vec2 initializes its accumulator on the first KV tile and does not consume slotCoeff. Avoid updating
            // empty online state.
            Adds(newMax, rowMaxFloatUb_, 0.0f, rowCount);
        } else {
            Max(newMax, oldMax, rowMaxFloatUb_, rowCount);
            PipeBarrier<PIPE_V>();
            Sub(slotCoeff, oldMax, newMax, rowCount);
            PipeBarrier<PIPE_V>();
            Exp(slotCoeff, slotCoeff, rowCount);
            Cast(rowMaxUb_, newMax, RoundMode::CAST_NONE, rowCount);
        }
        PipeBarrier<PIPE_V>();
        Muls(rowMaxUb_, rowMaxUb_, static_cast<half>(-1.0f), rowCount);
        LocalTensor<half> qBroadcast = brcBuf_.Get<half>()[FP16_BLOCK_LANES];
        Brcb(qBroadcast, qScaleBuf_.Get<half>()[chunk], static_cast<uint8_t>(Align8(rowCount) / FP32_BLOCK_LANES),
             {1, VECTOR_REPEAT_BLOCKS});
        PipeBarrier<PIPE_V>();
        Brcb(rowBrcUb_, rowMaxUb_, static_cast<uint8_t>(Align8(rowCount) / FP32_BLOCK_LANES),
             {1, VECTOR_REPEAT_BLOCKS});
        PipeBarrier<PIPE_V>();

        // Fuse Q scaling with maximum subtraction; retain FP16 rounding.
        BinaryRepeatParams subParams(1, 0, 0, rowStride, 1, 1);
        if constexpr (D64_REUSE_MASK)
            SetVectorMask<half>(FP16_REPEAT_LANES);
        for (uint32_t seg = 0; seg < segments; ++seg) {
            if (HeadDimAlign() > Int8GqaTnd::QFA_LARGE_FP32_HEAD_DIM_THRESHOLD) {
                // Round Q dequantization before max subtraction for the larger-D path. Keeping the two FP16
                // operations explicit stabilizes the following static INT8 P rounding.
                BinaryRepeatParams scaleParams = subParams;
                scaleParams.src0BlkStride = 1;
                scaleParams.src0RepStride = rowStride;
                Mul<half, true>(scoreUb_[seg * FP16_REPEAT_LANES], scoreUb_[seg * FP16_REPEAT_LANES], qBroadcast,
                                FP16_REPEAT_LANES, static_cast<uint8_t>(rowCount), scaleParams);
                PipeBarrier<PIPE_V>();
                Add<half, !D64_REUSE_MASK>(scoreUb_[seg * FP16_REPEAT_LANES], scoreUb_[seg * FP16_REPEAT_LANES],
                                           rowBrcUb_, FP16_REPEAT_LANES, static_cast<uint8_t>(rowCount), scaleParams);
            } else {
                FusedMulAdd<half, !D64_REUSE_MASK>(scoreUb_[seg * FP16_REPEAT_LANES], qBroadcast, rowBrcUb_,
                                                   FP16_REPEAT_LANES, static_cast<uint8_t>(rowCount), subParams);
            }
        }
        PipeBarrier<PIPE_V>();
        // Restore KV tail masks after FMA, where a zero Q scale can produce (-inf)*0.
        MaskScoreTail(nSize, rowCount);
        if (NeedsCausalMask(chunk, rowCount, nSize)) {
            ApplyCausalMask(rowCount, chunk, nSize, static_cast<half>(-__builtin_inff()));
        }
        if constexpr (D64_REUSE_MASK) {
            SetVectorMask<half>(FP16_REPEAT_LANES);
            Exp<half, false>(scoreUb_, scoreUb_, static_cast<uint64_t>(0),
                             static_cast<uint8_t>(rowCount * VectorColumns() / FP16_REPEAT_LANES),
                             {1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS});
        } else {
            Exp(scoreUb_, scoreUb_, rowCount * VectorColumns());
        }
        PipeBarrier<PIPE_V>();

        // Quantize P before destructive row-sum reduction, fusing P*p_scale with INT8 conversion.
        // One repeat writes 128 INT8 bytes, reads 256 FP16 bytes, and reuses the same P scale.
        BinaryRepeatParams quantParams(1, 1, D64_P_BROADCAST ? 0U : 1U, FP16_REPEAT_LANES / QFA_INT8_DMA_BLOCK_BYTES,
                                       VECTOR_REPEAT_BLOCKS, 0);
        const uint32_t quantCount = rowCount * VectorColumns();
        for (uint32_t offset = 0; offset < quantCount; offset += MAX_VECTOR_REPEATS * FP16_REPEAT_LANES) {
            const uint32_t repeats = Min32(MAX_VECTOR_REPEATS, (quantCount - offset) / FP16_REPEAT_LANES);
            MulCast<int8_t, half, !D64_REUSE_MASK>(pUb_[offset], scoreUb_[offset], quantScaleUb_,
                                                   static_cast<uint64_t>(FP16_REPEAT_LANES),
                                                   static_cast<uint8_t>(repeats), quantParams);
        }
        // MulCast must finish reading scores before the in-place row reduction overwrites them.
        PipeBarrier<PIPE_V>();
        if constexpr (D96_PREFILL)
            PublishP(slot, chunk, Min32(VecRows(), rows_ - chunk));

        // FIA's 512-column row reduction sums four aligned 128-column segments in place and issues only one
        // WholeReduceSum per row.
        BinaryRepeatParams reduceParams(1, 1, 1, rowStride, rowStride, rowStride);
        if (segments == MAX_SEGMENTS) {
            // Match FIA's pairwise sum: (0+1), (2+3), then combine.
            Add<half, !D64_REUSE_MASK>(scoreUb_, scoreUb_, scoreUb_[FP16_REPEAT_LANES], FP16_REPEAT_LANES,
                                       static_cast<uint8_t>(rowCount), reduceParams);
            Add<half, !D64_REUSE_MASK>(scoreUb_[2U * FP16_REPEAT_LANES], scoreUb_[2U * FP16_REPEAT_LANES],
                                       scoreUb_[3U * FP16_REPEAT_LANES], FP16_REPEAT_LANES,
                                       static_cast<uint8_t>(rowCount), reduceParams);
            PipeBarrier<PIPE_V>();
            Add<half, !D64_REUSE_MASK>(scoreUb_, scoreUb_, scoreUb_[2U * FP16_REPEAT_LANES], FP16_REPEAT_LANES,
                                       static_cast<uint8_t>(rowCount), reduceParams);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t seg = 1; seg < segments; ++seg) {
                Add<half, !D64_REUSE_MASK>(scoreUb_, scoreUb_, scoreUb_[seg * FP16_REPEAT_LANES], FP16_REPEAT_LANES,
                                           static_cast<uint8_t>(rowCount), reduceParams);
                PipeBarrier<PIPE_V>();
            }
        }
        WholeReduceSum<half, !D64_REUSE_MASK>(rowSumUb_, scoreUb_, FP16_REPEAT_LANES, rowCount, 1, 1, rowStride);
        PipeBarrier<PIPE_V>();
        Cast(rowSumFloatUb_, rowSumUb_, RoundMode::CAST_NONE, rowCount);
        PipeBarrier<PIPE_V>();
        if (isFirst) {
            Adds(newSum, rowSumFloatUb_, 0.0f, rowCount);
        } else {
            Mul(newSum, oldSum, slotCoeff, rowCount);
            PipeBarrier<PIPE_V>();
            Add(newSum, newSum, rowSumFloatUb_, rowCount);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (D64_REUSE_MASK)
            ResetMask();
    }

    // Apply one FP32 coefficient per row, using the FIA repeat pattern. A2/A3 vmadd computes dst = dst * src0 +
    // src1. Preserve this operand order.
    template <bool DIVIDE, typename T>
    __aicore__ inline void OutputRowOperation(const LocalTensor<T>& acc, const LocalTensor<T>& pv,
                                              const LocalTensor<T>& coefficient, uint32_t rows)
    {
        constexpr uint32_t BLOCK = QFA_INT8_DMA_BLOCK_BYTES / sizeof(T), REPEAT = VECTOR_REPEAT_BYTES / sizeof(T);
        const uint32_t stride = HeadDimAlign() / BLOCK;
        const uint32_t width = sizeof(T) == sizeof(half) ? HeadDimAlign() : HeadDim();
        BinaryRepeatParams params(1, DIVIDE ? 1 : 0, DIVIDE ? 0 : 1, stride, DIVIDE ? stride : 1, DIVIDE ? 1 : stride);
        const uint32_t full = width / REPEAT * REPEAT;
        for (uint32_t column = 0U; column < full; column += REPEAT)
            OutputColumnOperation<DIVIDE>(acc[column], pv[column], coefficient, REPEAT, rows, params);
        if (full < width)
            OutputColumnOperation<DIVIDE>(acc[full], pv[full], coefficient, width - full, rows, params);
    }
    template <bool DIVIDE, typename T>
    __aicore__ inline void OutputColumnOperation(const LocalTensor<T>& acc, const LocalTensor<T>& pv,
                                                 const LocalTensor<T>& coefficient, uint32_t count, uint32_t rows,
                                                 const BinaryRepeatParams& params)
    {
        if constexpr (DIVIDE)
            Div(acc, acc, coefficient, count, rows, params);
        else
            FusedMulAdd(acc, coefficient, pv, count, rows, params);
    }

    __aicore__ inline void CompactSgGroups(uint32_t start, uint32_t count, uint32_t tokens)
    {
        // One 32-byte DMA block per token contains G valid FP32 scales. A compact bitmap removes the padding and
        // the first/last fragments.
        auto bitmap = tmpBuf_.Get<uint8_t>()[SG_SCALE_BITMAP_OFFSET_BYTES].ReinterpretCast<uint32_t>();
        const uint32_t end = ((start + count - 1U) / tiling_.groupSize) * FP32_BLOCK_LANES +
                             (start + count - 1U) % tiling_.groupSize + 1U;
        const uint32_t words = (tokens * FP32_BLOCK_LANES + CAUSAL_MASK_WORD_BITS - 1U) / CAUSAL_MASK_WORD_BITS;
        const uint32_t pattern = ((1U << tiling_.groupSize) - 1U) * SG_SCALE_MASK_BYTE_BROADCAST;
        QfaSyncEvent<HardEvent::V_S>(EVENT_ID6);
        for (uint32_t i = 0; i < words; ++i) {
            uint32_t bits = pattern;
            if (i == 0U)
                bits &= UINT32_MAX << start;
            if (end < (i + 1U) * CAUSAL_MASK_WORD_BITS)
                bits &= end <= i * CAUSAL_MASK_WORD_BITS ? 0U : (1U << (end - i * CAUSAL_MASK_WORD_BITS)) - 1U;
            bitmap.SetValue(i, bits);
        }
        QfaSyncEvent<HardEvent::S_V>(EVENT_ID6);
        uint64_t selected = 0U;
        GatherMask(tmpBuf_.Get<float>(),
                   tmpBuf_.Get<uint8_t>()[STRIDED_SCALE_TMP_OFFSET_BYTES].ReinterpretCast<float>(), bitmap, true,
                   tokens * FP32_BLOCK_LANES, {1U, 1U, 0U, 0U}, selected);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherSgScales(uint32_t groupStart, uint32_t count)
    {
        auto index = tmpBuf_.Get<uint8_t>()[SG_SCALE_GATHER_INDEX_OFFSET_BYTES].ReinterpretCast<int32_t>();
        auto numer = tmpBuf_.Get<uint8_t>()[SG_SCALE_GATHER_NUMERATOR_OFFSET_BYTES].ReinterpretCast<float>();
        auto denom = tmpBuf_.Get<uint8_t>()[SG_SCALE_GATHER_DENOMINATOR_OFFSET_BYTES].ReinterpretCast<float>();
        auto quotient = denom.ReinterpretCast<int32_t>();
        const uint32_t n = Align8(count);
        // Build only the task's offset from immutable indices. Reusing the P/PV scratch must not introduce scalar
        // index stores between tasks.
        const auto qIndex = qScaleBuf_.Get<uint8_t>()[MAX_TASK_ROWS * sizeof(half)].ReinterpretCast<int32_t>();
        Adds(index, qIndex, static_cast<int32_t>(groupStart), n);
        PipeBarrier<PIPE_V>();
        if (tiling_.numHeadsKv > 1U) {
            if (tiling_.groupSize == 1U) {
                Muls(index, index, static_cast<int32_t>(tiling_.numHeadsQ), n);
            } else {
                Cast(numer, index, RoundMode::CAST_NONE, n);
                Duplicate(quotient, static_cast<int32_t>(tiling_.groupSize), n);
                PipeBarrier<PIPE_V>();
                Cast(denom, quotient, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
                Div(numer, numer, denom, n);
                PipeBarrier<PIPE_V>();
                // The bulk Q-scale path bounds the number of elements. This tolerance corrects reciprocal rounding at
                // integer boundaries, while staying below half the minimum distance between adjacent quotients.
                Adds(numer, numer, SG_SCALE_GATHER_FLOOR_EPSILON, n);
                PipeBarrier<PIPE_V>();
                Cast(quotient, numer, RoundMode::CAST_FLOOR, n);
                PipeBarrier<PIPE_V>();
                Muls(quotient, quotient, static_cast<int32_t>(tiling_.numHeadsQ - tiling_.groupSize), n);
                PipeBarrier<PIPE_V>();
                Add(index, index, quotient, n);
            }
            PipeBarrier<PIPE_V>();
            Adds(index, index, static_cast<int32_t>(kvHead_ * tiling_.groupSize), n);
            PipeBarrier<PIPE_V>();
        }
        ShiftLeft(index.ReinterpretCast<uint32_t>(), index.ReinterpretCast<uint32_t>(), FP32_BYTE_OFFSET_SHIFT, n);
        PipeBarrier<PIPE_V>();
        Gather(tmpBuf_.Get<float>(), tmpBuf_.Get<uint8_t>()[STRIDED_SCALE_TMP_OFFSET_BYTES].ReinterpretCast<float>(),
               index.ReinterpretCast<uint32_t>(), 0U, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompactStridedScales(LocalTensor<float> dst, uint32_t count)
    {
        // Reduce the padded DMA blocks to compact FP32 scales without aliasing K's Gather scratch.
        BlockReduceSum(dst, tmpBuf_.Get<uint8_t>()[STRIDED_SCALE_TMP_OFFSET_BYTES].ReinterpretCast<float>(),
                       static_cast<uint8_t>(Align8(count) / FP32_BLOCK_LANES), VECTOR_REPEAT_BYTES / sizeof(float), 1U,
                       1U, VECTOR_REPEAT_BLOCKS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline Int8GqaTnd::WindowRange RowWindow(uint32_t row) const
    {
        const uint32_t token = QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, mStart_ + rowStart_ + row);
        const int32_t right = tiling_.maskMode == 0U ? tiling_.winRight : 0;
        return Int8GqaTnd::QfaWindowRow(token, tiling_.seqLenQ, tiling_.seqLenKv, tiling_.winLeft, right);
    }

    __aicore__ inline void MaskSpan(uint32_t row, uint32_t run, uint32_t first, uint32_t end, half fill)
    {
        if (first == end)
            return;
        if ((first & TAIL_MASK_ALIGNMENT_BITS) != 0U) {
            const uint32_t block = first & ~TAIL_MASK_ALIGNMENT_BITS;
            const uint32_t finish = Min32(end, block + TAIL_MASK_LANES);
            uint64_t bits[VECTOR_MASK_WORDS] = {
                ((1ULL << (finish - block)) - 1ULL) & ~((1ULL << (first - block)) - 1ULL), 0ULL};
            Duplicate(scoreUb_[row * VectorColumns() + block], fill, bits, static_cast<uint8_t>(run), 1U,
                      VectorColumns() / FP16_BLOCK_LANES);
            first = finish;
        }
        for (uint32_t col = first; col < end; col += FP16_REPEAT_LANES) {
            Duplicate(scoreUb_[row * VectorColumns() + col], fill, Min32(FP16_REPEAT_LANES, end - col),
                      static_cast<uint8_t>(run), 1U, VectorColumns() / FP16_BLOCK_LANES);
        }
    }

    __aicore__ inline void ApplyWindowMask(uint32_t rows, uint32_t chunk, uint32_t nSize, half fill)
    {
        for (uint32_t r = 0U; r < rows;) {
            const uint32_t run =
                tiling_.layoutQ == Int8GqaTnd::QFA_LAYOUT_BNSD ?
                    1U :
                    QuantFlashAttnLayoutInt8<PATH>::QfaQueryRun(tiling_, mStart_ + rowStart_ + chunk + r, rows - r);
            const auto bounds = Int8GqaTnd::QfaWindowTile(RowWindow(chunk + r), nStart_, nSize);
            if (bounds.begin == bounds.end) {
                Duplicate(scoreUb_[r * VectorColumns()], fill, run * VectorColumns());
            } else {
                MaskSpan(r, run, 0U, bounds.begin, fill);
                MaskSpan(r, run, bounds.end, VectorColumns(), fill);
            }
            r += run;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool NeedsCausalMask(uint32_t chunk, uint32_t rows, uint32_t nSize) const
    {
        if constexpr (WINDOWED)
            return true;
        if (tiling_.maskMode == 0U)
            return false;
        const uint32_t start = mStart_ + rowStart_ + chunk;
        const uint32_t first = QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, start);
        const uint32_t minimum =
            QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) || first + rows <= tiling_.seqLenQ ? first : 0U;
        return static_cast<int64_t>(minimum) + 1 + tiling_.seqLenKv - tiling_.seqLenQ <
               static_cast<int64_t>(nStart_) + nSize;
    }

    __aicore__ inline uint32_t CausalColumns(uint32_t row, uint32_t nSize) const
    {
        if constexpr (WINDOWED) {
            const auto bounds = Int8GqaTnd::QfaWindowTile(RowWindow(row), nStart_, nSize);
            return bounds.end - bounds.begin;
        }
        const int64_t token = QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, mStart_ + rowStart_ + row);
        const int64_t count = token + 1 + static_cast<int64_t>(tiling_.seqLenKv) - tiling_.seqLenQ - nStart_;
        return count <= 0 ? 0U : (count < nSize ? static_cast<uint32_t>(count) : nSize);
    }

    __aicore__ inline void CreateMaskIndices(LocalTensor<int32_t> index, uint32_t count)
    {
        // Scalar CreateVecIndex seeds alias the still-live P/max scratch. Build each mask index block from the
        // immutable startup row indices.
        const auto rows = qScaleBuf_.Get<uint8_t>()[MAX_TASK_ROWS * sizeof(half)].template ReinterpretCast<int32_t>();
        for (uint32_t offset = 0U; offset < count; offset += MAX_TASK_ROWS)
            Adds(index[offset], rows, static_cast<int32_t>(offset), Min32(MAX_TASK_ROWS, count - offset));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyCausalMask(uint32_t rows, uint32_t chunk, uint32_t nSize, half fill)
    {
        if constexpr (WINDOWED) {
            ApplyWindowMask(rows, chunk, nSize, fill);
            return;
        }

        if constexpr (D96_PREFILL && PATH == Int8GqaPath::BSND) {
            // Keep GQA and every partial N tile on the existing mask path.
            if (tiling_.groupSize == 1U && tiling_.maskMode == Int8GqaTnd::QFA_MASK_CAUSAL_TILING &&
                tiling_.seqLenQ == tiling_.seqLenKv && tiling_.blockN == MAX_BLOCK_N && nSize == MAX_BLOCK_N &&
                VectorColumns() == MAX_BLOCK_N) {
                // Each bitmap word selects 32 scores in a complete N512 tile.
                const uint32_t wordsPerRow = MAX_BLOCK_N / CAUSAL_MASK_WORD_BITS;
                const uint32_t words = rows * wordsPerRow;
                auto index = pUb_.template ReinterpretCast<int32_t>();
                auto row = pUb_[P_MASK_ROW_OFFSET_BYTES].template ReinterpretCast<int32_t>();
                auto bitmap = pUb_[P_MASK_BITMAP_OFFSET_BYTES].template ReinterpretCast<uint32_t>();
                auto lut =
                    rowSumFloatBuf_.Get<uint8_t>()[CAUSAL_MASK_LUT_OFFSET_BYTES].template ReinterpretCast<uint32_t>();
                const int32_t validBase =
                    static_cast<int32_t>(static_cast<int64_t>(QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(
                                             tiling_, mStart_ + rowStart_ + chunk)) +
                                         1 + static_cast<int64_t>(tiling_.seqLenKv) - tiling_.seqLenQ - nStart_);
                CreateMaskIndices(index, words);
                // N512 has 16 words per row: shift by 4 for the row and keep the low 4 bits for the word.
                ShiftRight(row.template ReinterpretCast<uint32_t>(), index.template ReinterpretCast<uint32_t>(), 4U,
                           words);
                PipeBarrier<PIPE_V>();
                Adds(row, row, validBase, words);
                PipeBarrier<PIPE_V>();
                ShiftLeft(index.template ReinterpretCast<uint32_t>(), index.template ReinterpretCast<uint32_t>(), 28U,
                          words);
                PipeBarrier<PIPE_V>();
                ShiftRight(index.template ReinterpretCast<uint32_t>(), index.template ReinterpretCast<uint32_t>(), 28U,
                           words);
                PipeBarrier<PIPE_V>();
                // Shift by 5 to convert a word index to a 32-score offset.
                ShiftLeft(index.template ReinterpretCast<uint32_t>(), index.template ReinterpretCast<uint32_t>(), 5U,
                          words);
                PipeBarrier<PIPE_V>();
                Sub(index, row, index, words);
                PipeBarrier<PIPE_V>();
                Maxs(index, index, static_cast<int32_t>(0), words);
                PipeBarrier<PIPE_V>();
                Mins(index, index, static_cast<int32_t>(CAUSAL_MASK_WORD_BITS), words);
                PipeBarrier<PIPE_V>();
                ShiftLeft(index.template ReinterpretCast<uint32_t>(), index.template ReinterpretCast<uint32_t>(),
                          FP32_BYTE_OFFSET_SHIFT, words);
                PipeBarrier<PIPE_V>();
                Gather(bitmap, lut, index.template ReinterpretCast<uint32_t>(), 0U, words);
                PipeBarrier<PIPE_V>();
                // Feed CMPMASK from dead P scratch, avoiding Select's reserved-UB scalar source.
                auto fillUb = pUb_[P_MASK_FILL_OFFSET_BYTES].template ReinterpretCast<half>();
                Duplicate(fillUb, fill, FP16_BLOCK_LANES);
                PipeBarrier<PIPE_V>();
                SetCmpMask(fillUb);
                PipeBarrier<PIPE_V>();
                SetVectorMask<half>(FP16_REPEAT_LANES);
                BinaryRepeatParams selectParams;
                Select(scoreUb_, bitmap, scoreUb_, static_cast<uint8_t>(rows * MAX_SEGMENTS), selectParams);
                PipeBarrier<PIPE_V>();
                ResetMask();
                return;
            }
        }

        const uint32_t columns = VectorColumns();
        if constexpr (PATH != Int8GqaPath::BNSD) {
            if (QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) && tiling_.groupSize > 1U) {
                // Adjacent S1G heads share a causal boundary; repeat each mask over the whole run.
                const uint32_t firstRow = mStart_ + rowStart_ + chunk;
                uint32_t token = firstRow / tiling_.groupSize;
                uint32_t groupOffset = firstRow % tiling_.groupSize;
                for (uint32_t r = 0; r < rows;) {
                    const uint32_t run = Min32(rows - r, tiling_.groupSize - groupOffset);
                    const int64_t visible = static_cast<int64_t>(token) + 1 + static_cast<int64_t>(tiling_.seqLenKv) -
                                            tiling_.seqLenQ - nStart_;
                    const uint32_t valid =
                        visible <= 0 ? 0U : (visible < nSize ? static_cast<uint32_t>(visible) : nSize);
                    if (valid == 0U) {
                        Duplicate(scoreUb_[r * columns], fill, run * columns);
                    } else if (valid < columns) {
                        MaskSpan(r, run, valid, columns, fill);
                    }
                    r += run;
                    ++token;
                    groupOffset = 0U;
                }
                PipeBarrier<PIPE_V>();
                return;
            }
        }

        uint32_t token = QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, mStart_ + rowStart_ + chunk);
        const bool tokenMajor = QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ);
        for (uint32_t r = 0; r < rows; ++r) {
            const int64_t visible =
                static_cast<int64_t>(token) + 1 + static_cast<int64_t>(tiling_.seqLenKv) - tiling_.seqLenQ - nStart_;
            const uint32_t valid = visible <= 0 ? 0U : (visible < nSize ? static_cast<uint32_t>(visible) : nSize);
            const uint32_t aligned = (valid + TAIL_MASK_ALIGNMENT_BITS) & ~TAIL_MASK_ALIGNMENT_BITS;
            if (valid < aligned) {
                uint64_t mask[VECTOR_MASK_WORDS] = {
                    TAIL_MASK_ALL_LANES & ~((1ULL << (valid & TAIL_MASK_ALIGNMENT_BITS)) - 1ULL), 0ULL};
                Duplicate(scoreUb_[r * columns + (valid & ~TAIL_MASK_ALIGNMENT_BITS)], fill, mask, 1U, 1U,
                          VECTOR_REPEAT_BLOCKS);
            }
            if (aligned < columns) {
                Duplicate(scoreUb_[r * columns + aligned], fill, columns - aligned);
            }
            if (++token == tiling_.seqLenQ && !tokenMajor)
                token = 0U;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(uint32_t chunk, uint32_t chunkRows)
    {
        if constexpr (WINDOWED) {
            // Empty half-precision windows yield zero output; WriteLse already retained -inf.
            for (uint32_t row = 0U; row < chunkRows; ++row) {
                const auto window = RowWindow(chunk + row);
                if (window.begin == window.end)
                    Duplicate(pvUb_[row * HeadDimAlign()], static_cast<half>(0.0f), HeadDimAlign());
            }
            PipeBarrier<PIPE_V>();
        }
        if (tiling_.maskMode != 0U && tiling_.seqLenQ > tiling_.seqLenKv) {
            for (uint32_t row = 0; row < chunkRows; ++row) {
                const uint32_t queryRow = mStart_ + rowStart_ + chunk + row;
                const uint32_t token = PATH == Int8GqaPath::BNSD ?
                                           queryRow % tiling_.seqLenQ :
                                           QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, queryRow);
                if (token < tiling_.seqLenQ - tiling_.seqLenKv)
                    Duplicate(pvUb_[row * HeadDimAlign()], static_cast<half>(0.0f), HeadDimAlign());
            }
            PipeBarrier<PIPE_V>();
        }
        {
            // Only final writeback converts to BF16. Reuse the idle score-buffer tail, disjoint from PV and
            // broadcast scratch; no extra UB allocation.
            LocalTensor<float> castScratch =
                dataBuf_.Get<float>()[VecRows() * MAX_BLOCK_N - OUTPUT_CAST_CHUNK_ELEMENTS];
            const uint32_t castCount = chunkRows * HeadDimAlign();
            for (uint32_t offset = 0; offset < castCount; offset += OUTPUT_CAST_CHUNK_ELEMENTS) {
                const uint32_t size = Min32(castCount - offset, OUTPUT_CAST_CHUNK_ELEMENTS);
                Cast(castScratch, pvUb_[offset], RoundMode::CAST_NONE, size);
                PipeBarrier<PIPE_V>();
                Cast(pvUb_.ReinterpretCast<bfloat16_t>()[offset], castScratch, RoundMode::CAST_RINT, size);
                PipeBarrier<PIPE_V>();
            }
        }
        QfaSyncEvent<HardEvent::V_MTE3>(EVENT_ID0);
        if constexpr (PATH == Int8GqaPath::BNSD) {
            const uint64_t row =
                (static_cast<uint64_t>(batch_) * tiling_.numHeadsQ + kvHead_ * tiling_.groupSize) * tiling_.seqLenQ +
                mStart_ + rowStart_ + chunk;
            DataCopyExtParams p(chunkRows, OutputDim() * sizeof(half),
                                (HeadDimAlign() - ((OutputDim() + FP16_BLOCK_LANES - 1U) & ~(FP16_BLOCK_LANES - 1U))) *
                                    sizeof(half) / QFA_INT8_DMA_BLOCK_BYTES,
                                0U, 0U);
            DataCopyPad(outGm_[row * OutputDim()], pvUb_.ReinterpretCast<bfloat16_t>(), p);
        } else {
            for (uint32_t done = 0; done < chunkRows;) {
                const uint32_t row = mStart_ + rowStart_ + chunk + done;
                const uint32_t head = QuantFlashAttnLayoutInt8<PATH>::QfaQueryHead(tiling_, kvHead_, row);
                const uint32_t token = QuantFlashAttnLayoutInt8<PATH>::QfaQueryToken(tiling_, row);
                if (QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) &&
                    QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutOut) && OutputDim() == HeadDimAlign()) {
                    // S1G stores coalesce adjacent heads; Hkv=1 makes the entire chunk contiguous.
                    const uint32_t g = row % tiling_.groupSize;
                    uint32_t rows = Min32(chunkRows - done, tiling_.groupSize - g);
                    uint32_t blocks = 1U;
                    if (tiling_.numHeadsKv == 1U) {
                        rows = chunkRows - done;
                    } else if (g == 0U && chunkRows - done >= tiling_.groupSize) {
                        rows = tiling_.groupSize;
                        blocks = (chunkRows - done) / rows;
                    }
                    const uint64_t offset = QfaRow(tiling_.layoutOut, batch_, head, token, tiling_.numHeadsQ,
                                                   tiling_.seqLenQ, tiling_.qTokenStart) *
                                            OutputDim();
                    DataCopyExtParams group(
                        blocks, rows * OutputDim() * sizeof(half), 0U,
                        tiling_.numHeadsKv == 1U ? 0U : (tiling_.numHeadsQ - rows) * OutputDim() * sizeof(half), 0U);
                    DataCopyPad(outGm_[offset], pvUb_.template ReinterpretCast<bfloat16_t>()[done * HeadDimAlign()],
                                group);
                    done += blocks * rows;
                    continue;
                }
                const bool tokenRows = QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ) && tiling_.groupSize == 1U;
                const uint32_t count = tokenRows ?
                                           chunkRows - done :
                                           QuantFlashAttnLayoutInt8<PATH>::QfaQueryRun(tiling_, row, chunkRows - done);
                const uint64_t offset = QfaRow(tiling_.layoutOut, batch_, head, token, tiling_.numHeadsQ,
                                               tiling_.seqLenQ, tiling_.qTokenStart) *
                                        OutputDim();
                DataCopyExtParams p;
                p.blockCount = count;
                p.blockLen = OutputDim() * sizeof(half);
                p.srcStride = (HeadDimAlign() - ((OutputDim() + FP16_BLOCK_LANES - 1U) & ~(FP16_BLOCK_LANES - 1U))) *
                              sizeof(half) / QFA_INT8_DMA_BLOCK_BYTES;
                if (tokenRows) {
                    p.dstStride = tiling_.layoutOut == Int8GqaTnd::QFA_LAYOUT_BNSD ?
                                      0U :
                                      (tiling_.numHeadsQ - 1U) * OutputDim() * sizeof(half);
                } else if (QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ)) {
                    p.dstStride = tiling_.layoutOut == Int8GqaTnd::QFA_LAYOUT_BNSD ?
                                      (tiling_.seqLenQ - 1U) * OutputDim() * sizeof(half) :
                                      0U;
                } else {
                    p.dstStride = tiling_.layoutOut == Int8GqaTnd::QFA_LAYOUT_BNSD ?
                                      0U :
                                      (tiling_.numHeadsQ - 1U) * OutputDim() * sizeof(half);
                }
                DataCopyPad(outGm_[offset], pvUb_.template ReinterpretCast<bfloat16_t>()[done * HeadDimAlign()], p);
                done += count;
            }
        }
        QfaSyncEvent<HardEvent::MTE3_V>(EVENT_ID0);
        // Scalar scale-index stores must also follow compact output DMA.
        QfaSyncEvent<HardEvent::MTE3_S>(EVENT_ID0);
    }

    bool cacheK_ = false, largeFp32Accumulator_ = false, partitionedMhaHalf_ = false;
    uint32_t kCacheValid_ = 0U;
    uint32_t lastKBatch_ = Int8GqaTnd::QFA_INVALID_INDEX, lastKHead_ = Int8GqaTnd::QFA_INVALID_INDEX;
    uint32_t nStart_ = 0U;
    uint32_t vectorColumns_ = MAX_BLOCK_N;
    QuantFlashAttnTilingDataInt8 tiling_{};
    GlobalTensor<half> scoreGm_;
    GlobalTensor<int8_t> pGm_;
    GlobalTensor<half> pvGm_;
    GlobalTensor<bfloat16_t> outGm_;
    GlobalTensor<float> vDescaleGm_, kDescaleGm_, qDescaleGm_;
    GlobalTensor<uint64_t> kDeqWorkspaceGm_;
    TBuf<> dataBuf_, accBuf_, tmpBuf_, maxBuf_, sumBuf_, coeffBuf_, brcBuf_, quantScaleBuf_, rowMaxBuf_;
    TBuf<> rowSumBuf_, rowBrcBuf_, rowMaxFloatBuf_, rowSumFloatBuf_, kScaleIndexBuf_, qScaleBuf_;
    LocalTensor<half> scoreUb_;
    LocalTensor<int8_t> pUb_;
    LocalTensor<half> pvUb_;
    LocalTensor<float> maxStateUb_, sumStateUb_, coeffUb_, brcUb_;
    LocalTensor<half> quantScaleUb_, rowMaxUb_, rowSumUb_, rowBrcUb_;
    LocalTensor<float> rowMaxFloatUb_, rowSumFloatUb_;
    uint32_t subBlock_ = 0, batch_ = 0, kvHead_ = 0;
    uint32_t lastVHead_ = Int8GqaTnd::QFA_INVALID_INDEX;
    uint32_t mStart_ = 0, rowStart_ = 0, rows_ = 0, stateIdx_ = 0;
    bool haveAccum_ = false;
    uint32_t slotCount_ = Int8GqaTnd::QFA_PIPELINE_SLOTS;
    bool overlayPv_ = false;
    float pvFixInv_ = 1.0f;
    float pScaleValue_ = 0.0f;
};

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_VEC_INT8_H_
