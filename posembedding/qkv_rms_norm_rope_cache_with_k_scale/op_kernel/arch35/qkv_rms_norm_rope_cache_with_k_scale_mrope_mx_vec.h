/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VEC_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VEC_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_common.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_layout.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_struct.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vf.h"

namespace QkvRmsNormRopeCacheWithKScale {

/*
 * AIV-side M-RoPE MX pipeline. Persistent parameters and the gather index live
 * for the whole core range. Tile inputs use two UB slots. READY publishes
 * current contents to a consumer; FREE returns the same physical region to its
 * producer for overwrite. These are hardware-pipeline dependencies, not CPU
 * thread synchronization.
 *
 *   Resource/domain       Producer -> READY -> consumer       Last consumer -> FREE -> next producer
 *   qkv/rawCosSin slot n  MTE2 -> MTE2_V -> Vector V/K/Q      Vector Q -> V_MTE2 -> MTE2
 *   slotMapping slot n    MTE2 -> MTE2_S -> Scalar addressing released with the containing input slot
 *   position window       MTE2 -> MTE2_S -> Scalar gather     Scalar last tile -> S_MTE2 -> MTE2 refill
 *   V/K/Q staging         Vector -> V_MTE3 -> MTE3 store      MTE3 -> MTE3_V -> Vector
 *   MX scratch            Vector Store -> LocalMemBar -> Load final Load -> LocalMemBar -> next Store
 *
 * The input READY tokens cover the current tile only. V/K/Q staging READY and
 * FREE pairs are independent because each protects a different physical UB
 * region. Position uses a separate window lifetime and must not be folded into
 * the per-tile input-slot protocol.
 */
class QkvRmsNormRopeCacheWithKScaleMropeMxVec {
public:
    __aicore__ inline QkvRmsNormRopeCacheWithKScaleMropeMxVec(const GlobalTensors &tensors,
                                                              const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                                              uint64_t coreTokenBegin, uint64_t coreTokenEnd)
    {
        Init(tensors, tilingData, coreTokenBegin, coreTokenEnd);
        InitEvents();
    }

    __aicore__ inline ~QkvRmsNormRopeCacheWithKScaleMropeMxVec()
    {
        // Consume the final reuse tokens seeded by InitEvents or returned by the last tile.
        WaitFlag<HardEvent::V_MTE2>(INPUT_FREE_EVENT_0);
        WaitFlag<HardEvent::V_MTE2>(INPUT_FREE_EVENT_1);
        WaitFlag<HardEvent::MTE3_V>(V_FREE_EVENT);
        WaitFlag<HardEvent::MTE3_V>(K_FREE_EVENT);
        WaitFlag<HardEvent::MTE3_V>(Q_FREE_EVENT);
        WaitFlag<HardEvent::S_MTE2>(POSITION_FREE_EVENT);
    }

    QkvRmsNormRopeCacheWithKScaleMropeMxVec(const QkvRmsNormRopeCacheWithKScaleMropeMxVec &) = delete;
    QkvRmsNormRopeCacheWithKScaleMropeMxVec &operator=(const QkvRmsNormRopeCacheWithKScaleMropeMxVec &) = delete;
    QkvRmsNormRopeCacheWithKScaleMropeMxVec(QkvRmsNormRopeCacheWithKScaleMropeMxVec &&) = delete;
    QkvRmsNormRopeCacheWithKScaleMropeMxVec &operator=(QkvRmsNormRopeCacheWithKScaleMropeMxVec &&) = delete;

    __aicore__ inline void PrepareBeforeLoop()
    {
        // Publish the MTE2-loaded constants to Vector, then build the per-core gather index on Scalar and publish it.
        CopyPersistentInputs();
        SetFlag<HardEvent::MTE2_V>(PERSISTENT_READY_EVENT);
        WaitFlag<HardEvent::MTE2_V>(PERSISTENT_READY_EVENT);
        BuildMropeGatherIndex();
        SetFlag<HardEvent::S_V>(GATHER_READY_EVENT);
        WaitFlag<HardEvent::S_V>(GATHER_READY_EVENT);
    }

    __aicore__ inline void LoadTile(const MropeMxTileDesc &tile)
    {
        // FREE protects this slot from overwrite. slotMapping is published to Scalar separately from the QKV and
        // raw cos/sin data published to Vector because the two consumers run on different pipelines.
        WaitInputFree(tile.inputSlot);
        EnsurePositionWindow(tile);
        CopySlotMapping(tile);
        SetSlotReady(tile.inputSlot);
        CopyQkv(tile);
        GatherRawCosSin(tile);
        SetInputReady(tile.inputSlot);
    }

    __aicore__ inline void WaitTileReady(const MropeMxTileDesc &tile)
    {
        // Cache addressing and Vector computation may start only after both independent input domains are ready.
        WaitSlotReady(tile.inputSlot);
        WaitInputReady(tile.inputSlot);
    }

    __aicore__ inline void ProcessV(const MropeMxTileDesc &tile)
    {
        // V/K/Q staging buffers are single-buffered. Each FREE wait prevents Vector from overwriting a pending MTE3
        // write, and the matching READY event publishes the newly produced staging data to MTE3.
        WaitFlag<HardEvent::MTE3_V>(V_FREE_EVENT);
        // TND [tile.tokenCount, headCount_(qHeadNum_ + kvHeadNum_ + kvHeadNum_), headDim_]
        const uint32_t inputTokenStride = static_cast<uint32_t>(headCount_ * headDim_);
        const uint32_t vInputOffset = static_cast<uint32_t>((qHeadNum_ + kvHeadNum_) * headDim_);
        AscendC::VF_CALL<VScaleFp8D128ToNtdVfImpl<true>>(
            (__ubuf__ bfloat16_t *)inputQkvUb_[tile.inputSlot][vInputOffset].GetPhyAddr(),
            (__ubuf__ float *)vScaleUb_.GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)vDataUb_.GetPhyAddr(),
            static_cast<uint16_t>(tile.tokenCount), static_cast<uint16_t>(kvHeadNum_), inputTokenStride,
            static_cast<uint32_t>(headDim_));
        SetFlag<HardEvent::V_MTE3>(V_READY_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_READY_EVENT);
        ScatterVCache(tile);
        SetFlag<HardEvent::MTE3_V>(V_FREE_EVENT);
    }

    __aicore__ inline void ProcessK(const MropeMxTileDesc &tile)
    {
        WaitFlag<HardEvent::MTE3_V>(K_FREE_EVENT);
        const uint32_t inputTokenStride = static_cast<uint32_t>(headCount_ * headDim_);
        const uint32_t kInputOffset = static_cast<uint32_t>(qHeadNum_ * headDim_);
        const uint32_t scaleTokenStrideWords =
            MropeMxAlignDevice(static_cast<uint32_t>(kvHeadNum_ * MROPE_MX_SCALE_COUNT_D128)) / sizeof(uint32_t);
        __ubuf__ bfloat16_t *kInput = (__ubuf__ bfloat16_t *)inputQkvUb_[tile.inputSlot][kInputOffset].GetPhyAddr();
        __ubuf__ float *kGamma = (__ubuf__ float *)kGammaUb_.GetPhyAddr();
        __ubuf__ float *rawCosSin = (__ubuf__ float *)rawCosSinUb_[tile.inputSlot].GetPhyAddr();
        __ubuf__ uint32_t *gatherIndex = (__ubuf__ uint32_t *)gatherIndexUb_.GetPhyAddr();
        __ubuf__ fp8_e4m3fn_t *kData = (__ubuf__ fp8_e4m3fn_t *)kDataUb_.GetPhyAddr();
        __ubuf__ fp8_e8m0_t *kScale = (__ubuf__ fp8_e8m0_t *)kScaleUb_.GetPhyAddr();
        __ubuf__ uint8_t *mxScratch = (__ubuf__ uint8_t *)mxScratchUb_.GetPhyAddr();
        const uint16_t tokenCount = static_cast<uint16_t>(tile.tokenCount);
        const uint16_t kHeadCount = static_cast<uint16_t>(kvHeadNum_);
        const uint32_t inputHeadStride = static_cast<uint32_t>(headDim_);
        // Pack complete tokens into a 16-row VF. Nk=8 keeps all token pairs and a possible final
        // tail8 in one tile-wide pipeline; the other supported even divisors keep their existing schedule.
        if (kHeadCount == 8U) {
            AscendC::VF_CALL<QRmsNormMropeMxD128GlobalTileWaveVfImpl<true>>(
                kInput, kGamma, rawCosSin, gatherIndex, kData, kScale, mxScratch, tokenCount, inputTokenStride,
                kHeadCount, inputHeadStride, scaleTokenStrideWords, epsilon_);
        } else if ((kHeadCount & 1U) == 0U && (MROPE_MX_ROW_BATCH_ROWS % kHeadCount) == 0U) {
            const uint16_t tokensPerBatch = static_cast<uint16_t>(MROPE_MX_ROW_BATCH_ROWS / kHeadCount);
            const uint16_t batchTokenCount = static_cast<uint16_t>(tokenCount / tokensPerBatch * tokensPerBatch);
            if (batchTokenCount != 0U) {
                AscendC::VF_CALL<KRmsNormMropeMxD128RowBatch16EvenVfImpl>(
                    kInput, kGamma, rawCosSin, gatherIndex, kData, kScale, mxScratch, batchTokenCount, kHeadCount,
                    inputTokenStride, inputHeadStride, scaleTokenStrideWords, epsilon_);
            }
            if (batchTokenCount != tokenCount) {
                const uint16_t tailTokenCount = static_cast<uint16_t>(tokenCount - batchTokenCount);
                const uint32_t tailInputOffset = static_cast<uint32_t>(batchTokenCount) * inputTokenStride;
                const uint32_t tailRawOffset = static_cast<uint32_t>(batchTokenCount) * 3U * inputHeadStride;
                const uint32_t tailDataOffset = static_cast<uint32_t>(batchTokenCount) * kHeadCount * inputHeadStride;
                const uint32_t tailScaleOffset =
                    static_cast<uint32_t>(batchTokenCount) * scaleTokenStrideWords * sizeof(uint32_t);
                AscendC::VF_CALL<KRmsNormMropeMxD128VfImpl>(
                    kInput + tailInputOffset, kGamma, rawCosSin + tailRawOffset, gatherIndex, kData + tailDataOffset,
                    kScale + tailScaleOffset, mxScratch, tailTokenCount, kHeadCount, inputTokenStride, inputHeadStride,
                    scaleTokenStrideWords, epsilon_);
            }
        } else {
            AscendC::VF_CALL<KRmsNormMropeMxD128VfImpl>(kInput, kGamma, rawCosSin, gatherIndex, kData, kScale,
                                                        mxScratch, tokenCount, kHeadCount, inputTokenStride,
                                                        inputHeadStride, scaleTokenStrideWords, epsilon_);
        }
        SetFlag<HardEvent::V_MTE3>(K_READY_EVENT);
        WaitFlag<HardEvent::V_MTE3>(K_READY_EVENT);
        ScatterKCacheAndScale(tile);
        SetFlag<HardEvent::MTE3_V>(K_FREE_EVENT);
    }

    __aicore__ inline void ProcessQ(const MropeMxTileDesc &tile)
    {
        WaitFlag<HardEvent::MTE3_V>(Q_FREE_EVENT);
        const uint32_t inputTokenStride = static_cast<uint32_t>(headCount_ * headDim_);
        ProcessQGlobalTileWave(tile, inputTokenStride);
        SetFlag<HardEvent::V_MTE3>(Q_READY_EVENT);
        WaitFlag<HardEvent::V_MTE3>(Q_READY_EVENT);
        StoreQAndScale(tile);
        SetFlag<HardEvent::MTE3_V>(Q_FREE_EVENT);
    }

    __aicore__ inline void ProcessQGlobalTileWave(const MropeMxTileDesc &tile, uint32_t inputTokenStride)
    {
        if ((qHeadNum_ % MROPE_MX_ROW_BATCH_ROWS) == MROPE_MX_ROW_BATCH_ROWS / 2U) {
            AscendC::VF_CALL<QRmsNormMropeMxD128GlobalTileWaveVfImpl<true>>(
                (__ubuf__ bfloat16_t *)inputQkvUb_[tile.inputSlot].GetPhyAddr(),
                (__ubuf__ float *)qGammaUb_.GetPhyAddr(), (__ubuf__ float *)rawCosSinUb_[tile.inputSlot].GetPhyAddr(),
                (__ubuf__ uint32_t *)gatherIndexUb_.GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)qDataUb_.GetPhyAddr(),
                (__ubuf__ fp8_e8m0_t *)qScaleUb_.GetPhyAddr(), (__ubuf__ uint8_t *)mxScratchUb_.GetPhyAddr(),
                static_cast<uint16_t>(tile.tokenCount), inputTokenStride, static_cast<uint16_t>(qHeadNum_),
                static_cast<uint32_t>(headDim_), static_cast<uint32_t>(qHeadNum_), epsilon_);
        } else {
            AscendC::VF_CALL<QRmsNormMropeMxD128GlobalTileWaveVfImpl<false>>(
                (__ubuf__ bfloat16_t *)inputQkvUb_[tile.inputSlot].GetPhyAddr(),
                (__ubuf__ float *)qGammaUb_.GetPhyAddr(), (__ubuf__ float *)rawCosSinUb_[tile.inputSlot].GetPhyAddr(),
                (__ubuf__ uint32_t *)gatherIndexUb_.GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)qDataUb_.GetPhyAddr(),
                (__ubuf__ fp8_e8m0_t *)qScaleUb_.GetPhyAddr(), (__ubuf__ uint8_t *)mxScratchUb_.GetPhyAddr(),
                static_cast<uint16_t>(tile.tokenCount), inputTokenStride, static_cast<uint16_t>(qHeadNum_),
                static_cast<uint32_t>(headDim_), static_cast<uint32_t>(qHeadNum_), epsilon_);
        }
    }

    // Q is the last consumer of inputQkv/rawCosSin, so only the controller's post-Q release may return the slot.
    __aicore__ inline void ReleaseInput(const MropeMxTileDesc &tile)
    {
        SetInputFree(tile.inputSlot);
    }

private:
    // Event IDs are scoped by HardEvent. READY means the consumer may read current contents; FREE means the producer
    // may overwrite the physical buffer. INPUT and SLOT are per input slot; POSITION and V/K/Q protect shared UB.
    static constexpr event_t INPUT_FREE_EVENT_0 = static_cast<event_t>(0U);
    static constexpr event_t INPUT_FREE_EVENT_1 = static_cast<event_t>(1U);
    static constexpr event_t INPUT_READY_EVENT_0 = static_cast<event_t>(0U);
    static constexpr event_t INPUT_READY_EVENT_1 = static_cast<event_t>(1U);
    static constexpr event_t SLOT_READY_EVENT_0 = static_cast<event_t>(0U);
    static constexpr event_t SLOT_READY_EVENT_1 = static_cast<event_t>(1U);
    static constexpr event_t POSITION_FREE_EVENT = static_cast<event_t>(2U);
    static constexpr event_t POSITION_READY_EVENT = static_cast<event_t>(2U);
    static constexpr event_t V_FREE_EVENT = static_cast<event_t>(0U);
    static constexpr event_t K_FREE_EVENT = static_cast<event_t>(1U);
    static constexpr event_t Q_FREE_EVENT = static_cast<event_t>(2U);
    static constexpr event_t V_READY_EVENT = static_cast<event_t>(0U);
    static constexpr event_t K_READY_EVENT = static_cast<event_t>(1U);
    static constexpr event_t Q_READY_EVENT = static_cast<event_t>(2U);
    static constexpr event_t PERSISTENT_READY_EVENT = static_cast<event_t>(3U);
    static constexpr event_t GATHER_READY_EVENT = static_cast<event_t>(0U);

    __aicore__ inline void Init(const GlobalTensors &tensors, const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                uint64_t coreTokenBegin, uint64_t coreTokenEnd)
    {
        qHeadNum_ = tilingData->qHeadNum;
        kvHeadNum_ = tilingData->kvHeadNum;
        headCount_ = qHeadNum_ + kvHeadNum_ + kvHeadNum_;
        headDim_ = tilingData->headDim;
        blockSize_ = tilingData->blockSize;
        kvCacheStrideBlock_ = tilingData->kvCacheStrideBlock;
        kvCacheStrideHead_ = tilingData->kvCacheStrideHead;
        kvCacheStrideToken_ = tilingData->kvCacheStrideToken;
        kScaleCacheStrideBlock_ = tilingData->kScaleCacheStrideBlock;
        kScaleCacheStrideHead_ = tilingData->kScaleCacheStrideHead;
        kScaleCacheStrideToken_ = tilingData->kScaleCacheStrideToken;
        epsilon_ = tilingData->epsilon;
        mropeSectionH_ = tilingData->mropeSectionH;
        mropeSectionW_ = tilingData->mropeSectionW;
        tokenTile_ = tilingData->tokenTile;
        coreTokenBegin_ = coreTokenBegin;
        coreTokenEnd_ = coreTokenEnd;
        positionWindowBegin_ = coreTokenBegin;
        positionWindowEnd_ = coreTokenBegin;
        MakeMropeMxQGlobalTileWaveUbLayoutDevice(static_cast<uint32_t>(tilingData->tokenTile),
                                                 static_cast<uint32_t>(qHeadNum_), static_cast<uint32_t>(kvHeadNum_),
                                                 static_cast<uint32_t>(headDim_), layout_);
        BindGlobalTensors(tensors);
        BindLocalTensors(layout_);
    }

    __aicore__ inline void InitEvents()
    {
        // No consumer owns a reusable buffer initially, so seed one FREE token for every physical resource.
        SetFlag<HardEvent::V_MTE2>(INPUT_FREE_EVENT_0);
        SetFlag<HardEvent::V_MTE2>(INPUT_FREE_EVENT_1);
        SetFlag<HardEvent::MTE3_V>(V_FREE_EVENT);
        SetFlag<HardEvent::MTE3_V>(K_FREE_EVENT);
        SetFlag<HardEvent::MTE3_V>(Q_FREE_EVENT);
        SetFlag<HardEvent::S_MTE2>(POSITION_FREE_EVENT);
    }

    __aicore__ inline void WaitInputFree(uint32_t slot)
    {
        WaitFlag<HardEvent::V_MTE2>(slot == 0U ? INPUT_FREE_EVENT_0 : INPUT_FREE_EVENT_1);
    }
    __aicore__ inline void SetInputReady(uint32_t slot)
    {
        SetFlag<HardEvent::MTE2_V>(slot == 0U ? INPUT_READY_EVENT_0 : INPUT_READY_EVENT_1);
    }
    __aicore__ inline void WaitInputReady(uint32_t slot)
    {
        WaitFlag<HardEvent::MTE2_V>(slot == 0U ? INPUT_READY_EVENT_0 : INPUT_READY_EVENT_1);
    }
    __aicore__ inline void SetInputFree(uint32_t slot)
    {
        SetFlag<HardEvent::V_MTE2>(slot == 0U ? INPUT_FREE_EVENT_0 : INPUT_FREE_EVENT_1);
    }
    __aicore__ inline void SetSlotReady(uint32_t slot)
    {
        SetFlag<HardEvent::MTE2_S>(slot == 0U ? SLOT_READY_EVENT_0 : SLOT_READY_EVENT_1);
    }
    __aicore__ inline void WaitSlotReady(uint32_t slot)
    {
        WaitFlag<HardEvent::MTE2_S>(slot == 0U ? SLOT_READY_EVENT_0 : SLOT_READY_EVENT_1);
    }

    __aicore__ inline void CopyPersistentInputs()
    {
        DataCopyGmToUb2D(qGammaUb_, qGammaGm_, 1U, headDim_, headDim_);
        DataCopyGmToUb2D(kGammaUb_, kGammaGm_, 1U, headDim_, headDim_);
        DataCopyGmToUb2D(vScaleUb_, vScaleGm_, 1U, kvHeadNum_ * headDim_, kvHeadNum_ * headDim_);
    }

    __aicore__ inline void BuildMropeGatherIndex()
    {
        // M-RoPE interleaves temporal/height/width lanes in groups of three. Each entry selects one axis row from
        // rawCosSin [3, D] while preserving the lane offset within that row.
        for (uint32_t lane = 0U; lane < QKV_K_SCALE_D128_HALF_SIZE; ++lane) {
            const uint32_t repeat = lane / 3U;
            const uint32_t axisInGroup = lane % 3U;
            uint32_t axis = 0U;
            if (axisInGroup == 1U && repeat < mropeSectionH_) {
                axis = 1U;
            } else if (axisInGroup == 2U && repeat < mropeSectionW_) {
                axis = 2U;
            }
            gatherIndexUb_.SetValue(lane, axis * QKV_K_SCALE_D128_FULL_SIZE + lane);
        }
    }

    __aicore__ inline void EnsurePositionWindow(const MropeMxTileDesc &tile)
    {
        // Keep a bounded position table in UB and refill it only when the current tile leaves the resident window.
        const uint64_t tileEnd = tile.tokenBegin + tile.tokenCount;
        if (tile.tokenBegin >= positionWindowBegin_ && tileEnd <= positionWindowEnd_) {
            return;
        }
        const uint64_t relative = tile.tokenBegin - coreTokenBegin_;
        const uint64_t windowBegin =
            coreTokenBegin_ + relative / MROPE_MX_POSITION_WINDOW_TOKENS * MROPE_MX_POSITION_WINDOW_TOKENS;
        const uint32_t windowCount =
            static_cast<uint32_t>(MinU64(windowBegin + MROPE_MX_POSITION_WINDOW_TOKENS, coreTokenEnd_) - windowBegin);
        LoadPositionWindow(windowBegin, windowCount);
    }

    __aicore__ inline void LoadPositionWindow(uint64_t windowBegin, uint32_t windowTokenCount)
    {
        // Scalar returns POSITION_FREE after gathering the window's last tile; MTE2 must not refill it earlier.
        WaitFlag<HardEvent::S_MTE2>(POSITION_FREE_EVENT);
        DataCopyGmToUb2D(positionUb_, mropePositionGm_[windowBegin * 3U], 1U,
                         static_cast<uint64_t>(windowTokenCount) * 3U, static_cast<uint64_t>(windowTokenCount) * 3U);
        SetFlag<HardEvent::MTE2_S>(POSITION_READY_EVENT);
        WaitFlag<HardEvent::MTE2_S>(POSITION_READY_EVENT);
        positionWindowBegin_ = windowBegin;
        positionWindowEnd_ = windowBegin + windowTokenCount;
    }

    __aicore__ inline void CopySlotMapping(const MropeMxTileDesc &tile)
    {
        DataCopyGmToUb2D(slotMappingUb_[tile.inputSlot], slotMappingGm_[tile.tokenBegin], 1U, tile.tokenCount,
                         tile.tokenCount);
    }

    __aicore__ inline void CopyQkv(const MropeMxTileDesc &tile)
    {
        const uint64_t elements = static_cast<uint64_t>(tile.tokenCount) * headCount_ * headDim_;
        DataCopyGmToUb2D(inputQkvUb_[tile.inputSlot], qkvGm_[tile.tokenBegin * headCount_ * headDim_], 1U, elements,
                         elements);
    }

    __aicore__ inline void GatherRawCosSin(const MropeMxTileDesc &tile)
    {
        // Materialize the three position-selected cos/sin rows per token so the VF can gather all M-RoPE lanes in UB.
        const uint64_t positionOffset = tile.tokenBegin - positionWindowBegin_;
        for (uint32_t tokenIdx = 0U; tokenIdx < tile.tokenCount; ++tokenIdx) {
            for (uint32_t axis = 0U; axis < 3U; ++axis) {
                const int32_t position = positionUb_.GetValue((positionOffset + tokenIdx) * 3U + axis);
                const uint64_t sourceOffset = static_cast<uint64_t>(position) * headDim_;
                const uint64_t destinationOffset = (static_cast<uint64_t>(tokenIdx) * 3U + axis) * headDim_;
                DataCopyGmToUb2D(rawCosSinUb_[tile.inputSlot][destinationOffset], cosSinGm_[sourceOffset], 1U, headDim_,
                                 headDim_);
            }
        }
        if (tile.tokenBegin + tile.tokenCount == positionWindowEnd_) {
            // This tile is the last Scalar reader of the resident position window.
            SetFlag<HardEvent::S_MTE2>(POSITION_FREE_EVENT);
        }
    }

    __aicore__ inline uint64_t CacheDataOffset(const MropeMxTileDesc &tile, uint32_t tokenIdx) const
    {
        const uint64_t slot = static_cast<uint64_t>(slotMappingUb_[tile.inputSlot].GetValue(tokenIdx));
        return slot / blockSize_ * kvCacheStrideBlock_ + slot % blockSize_ * kvCacheStrideToken_;
    }

    __aicore__ inline uint64_t CacheScaleOffset(const MropeMxTileDesc &tile, uint32_t tokenIdx) const
    {
        const uint64_t slot = static_cast<uint64_t>(slotMappingUb_[tile.inputSlot].GetValue(tokenIdx));
        return slot / blockSize_ * kScaleCacheStrideBlock_ + slot % blockSize_ * kScaleCacheStrideToken_;
    }

    __aicore__ inline void ScatterVCache(const MropeMxTileDesc &tile)
    {
        // VScaleFp8D128ToNtdVfImpl stores [Nv, tileT, D].  For one cache
        // token, walk heads with the VF's head stride instead of interpreting
        // the staging buffer as [tileT, Nv, D].
        const uint64_t ubHeadStride = static_cast<uint64_t>(tile.tokenCount) * headDim_;
        for (uint32_t tokenIdx = 0U; tokenIdx < tile.tokenCount; ++tokenIdx) {
            DataCopyUbToGm2D(vCacheOutGm_[CacheDataOffset(tile, tokenIdx)],
                             vDataUb_[static_cast<uint64_t>(tokenIdx) * headDim_], kvHeadNum_, headDim_, ubHeadStride,
                             kvCacheStrideHead_);
        }
    }

    __aicore__ inline void ScatterKCacheAndScale(const MropeMxTileDesc &tile)
    {
        const uint64_t dataTokenStride = kvHeadNum_ * headDim_;
        const uint64_t scaleTokenStride =
            MropeMxAlignDevice(static_cast<uint32_t>(kvHeadNum_ * MROPE_MX_SCALE_COUNT_D128 * MROPE_MX_FP8_BYTES));
        for (uint32_t tokenIdx = 0U; tokenIdx < tile.tokenCount; ++tokenIdx) {
            DataCopyUbToGm2D(kCacheOutGm_[CacheDataOffset(tile, tokenIdx)],
                             kDataUb_[static_cast<uint64_t>(tokenIdx) * dataTokenStride], kvHeadNum_, headDim_,
                             headDim_, kvCacheStrideHead_);
            DataCopyScaleCompact(kScaleCacheOutGm_[CacheScaleOffset(tile, tokenIdx)],
                                 kScaleUb_[static_cast<uint64_t>(tokenIdx) * scaleTokenStride], kvHeadNum_,
                                 kScaleCacheStrideHead_);
        }
    }

    __aicore__ inline void StoreQAndScale(const MropeMxTileDesc &tile)
    {
        const uint64_t dataElements = static_cast<uint64_t>(tile.tokenCount) * qHeadNum_ * headDim_;
        const uint64_t scaleElements = static_cast<uint64_t>(tile.tokenCount) * qHeadNum_ * MROPE_MX_SCALE_COUNT_D128;
        DataCopyUbToGm2D(qOutGm_[tile.tokenBegin * qHeadNum_ * headDim_], qDataUb_, 1U, dataElements, dataElements,
                         dataElements);
        DataCopyScaleContiguous(qScaleGm_[tile.tokenBegin * qHeadNum_ * MROPE_MX_SCALE_COUNT_D128], qScaleUb_,
                                scaleElements);
    }

    __aicore__ inline void DataCopyScaleCompact(const GlobalTensor<fp8_e8m0_t> &dst, const LocalTensor<fp8_e8m0_t> &src,
                                                uint64_t rowCount, uint64_t dstStride)
    {
        // Reinterpret as uint8_t because DataCopyExtParams lengths and strides below are expressed in bytes.
        DataCopyExtParams params{static_cast<uint16_t>(rowCount), static_cast<uint32_t>(MROPE_MX_SCALE_COUNT_D128), 0U,
                                 static_cast<uint32_t>(dstStride - MROPE_MX_SCALE_COUNT_D128), 0U};
        DataCopyPad<uint8_t, AscendC::PaddingMode::Compact>(dst.template ReinterpretCast<uint8_t>(),
                                                            src.template ReinterpretCast<uint8_t>(), params);
    }

    __aicore__ inline void DataCopyScaleContiguous(const GlobalTensor<fp8_e8m0_t> &dst,
                                                   const LocalTensor<fp8_e8m0_t> &src, uint64_t elementCount)
    {
        // One E8M0 element is one byte, so elementCount is also the transfer byte count.
        DataCopyExtParams params{1U, static_cast<uint32_t>(elementCount), 0U, 0U, 0U};
        DataCopyPad<uint8_t, AscendC::PaddingMode::Compact>(dst.template ReinterpretCast<uint8_t>(),
                                                            src.template ReinterpretCast<uint8_t>(), params);
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        qkvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qkv);
        qkvGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        qGammaGm_.SetGlobalBuffer((__gm__ float *)tensors.qGamma);
        kGammaGm_.SetGlobalBuffer((__gm__ float *)tensors.kGamma);
        cosSinGm_.SetGlobalBuffer((__gm__ float *)tensors.cosSin);
        slotMappingGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.slotMapping);
        vScaleGm_.SetGlobalBuffer((__gm__ float *)tensors.vScale);
        mropePositionGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.mropePosition);
        qOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.qOut);
        qScaleGm_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)tensors.qScale);
        kCacheOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCacheOut);
        vCacheOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCacheOut);
        kScaleCacheOutGm_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)tensors.kScaleCacheOut);
    }

    __aicore__ inline void BindLocalTensors(const MropeMxUbLayout &layout)
    {
        // Only tile inputs are double-buffered. V/K/Q outputs and MX scratch are shared and protected by FREE events.
        qGammaUb_ = LocalTensor<float>(TPosition::LCM, layout.qGammaOffsetBytes, headDim_);
        kGammaUb_ = LocalTensor<float>(TPosition::LCM, layout.kGammaOffsetBytes, headDim_);
        vScaleUb_ = LocalTensor<float>(TPosition::LCM, layout.vScaleOffsetBytes, kvHeadNum_ * headDim_);
        gatherIndexUb_ = LocalTensor<uint32_t>(TPosition::LCM, layout.gatherIndexOffsetBytes, headDim_ / 2U);
        positionUb_ =
            LocalTensor<int32_t>(TPosition::LCM, layout.positionOffsetBytes, MROPE_MX_POSITION_WINDOW_TOKENS * 3U);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            inputQkvUb_[slot] = LocalTensor<bfloat16_t>(TPosition::LCM, layout.inputQkvOffsetBytes[slot],
                                                        tokenTile_ * headCount_ * headDim_);
            rawCosSinUb_[slot] =
                LocalTensor<float>(TPosition::LCM, layout.inputRawCosSinOffsetBytes[slot], tokenTile_ * 3U * headDim_);
            slotMappingUb_[slot] =
                LocalTensor<int32_t>(TPosition::LCM, layout.inputSlotMappingOffsetBytes[slot], tokenTile_);
        }
        vDataUb_ =
            LocalTensor<fp8_e4m3fn_t>(TPosition::LCM, layout.vDataOffsetBytes, tokenTile_ * kvHeadNum_ * headDim_);
        kDataUb_ =
            LocalTensor<fp8_e4m3fn_t>(TPosition::LCM, layout.kDataOffsetBytes, tokenTile_ * kvHeadNum_ * headDim_);
        const uint64_t kScaleTokenStride =
            MropeMxAlignDevice(static_cast<uint32_t>(kvHeadNum_ * MROPE_MX_SCALE_COUNT_D128 * MROPE_MX_FP8_BYTES));
        kScaleUb_ = LocalTensor<fp8_e8m0_t>(TPosition::LCM, layout.kScaleOffsetBytes, tokenTile_ * kScaleTokenStride);
        qDataUb_ =
            LocalTensor<fp8_e4m3fn_t>(TPosition::LCM, layout.qDataOffsetBytes, tokenTile_ * qHeadNum_ * headDim_);
        qScaleUb_ = LocalTensor<fp8_e8m0_t>(TPosition::LCM, layout.qScaleOffsetBytes,
                                            tokenTile_ * qHeadNum_ * MROPE_MX_SCALE_COUNT_D128);
        mxScratchUb_ = LocalTensor<uint8_t>(TPosition::LCM, layout.mxScratchOffsetBytes, layout.mxScratchBytes);
    }

    MropeMxUbLayout layout_{};
    uint64_t qHeadNum_ = 0U;
    uint64_t kvHeadNum_ = 0U;
    uint64_t headCount_ = 0U;
    uint64_t headDim_ = 0U;
    uint64_t blockSize_ = 0U;
    uint64_t kvCacheStrideBlock_ = 0U;
    uint64_t kvCacheStrideHead_ = 0U;
    uint64_t kvCacheStrideToken_ = 0U;
    uint64_t kScaleCacheStrideBlock_ = 0U;
    uint64_t kScaleCacheStrideHead_ = 0U;
    uint64_t kScaleCacheStrideToken_ = 0U;
    uint64_t mropeSectionH_ = 0U;
    uint64_t mropeSectionW_ = 0U;
    uint64_t tokenTile_ = 0U;
    uint64_t coreTokenBegin_ = 0U;
    uint64_t coreTokenEnd_ = 0U;
    uint64_t positionWindowBegin_ = 0U;
    uint64_t positionWindowEnd_ = 0U;
    float epsilon_ = 0.0F;

    GlobalTensor<bfloat16_t> qkvGm_;
    GlobalTensor<float> qGammaGm_;
    GlobalTensor<float> kGammaGm_;
    GlobalTensor<float> cosSinGm_;
    GlobalTensor<int32_t> slotMappingGm_;
    GlobalTensor<float> vScaleGm_;
    GlobalTensor<int32_t> mropePositionGm_;
    GlobalTensor<fp8_e4m3fn_t> qOutGm_;
    GlobalTensor<fp8_e8m0_t> qScaleGm_;
    GlobalTensor<fp8_e4m3fn_t> kCacheOutGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheOutGm_;
    GlobalTensor<fp8_e8m0_t> kScaleCacheOutGm_;

    LocalTensor<float> qGammaUb_;
    LocalTensor<float> kGammaUb_;
    LocalTensor<float> vScaleUb_;
    LocalTensor<uint32_t> gatherIndexUb_;
    LocalTensor<int32_t> positionUb_;
    LocalTensor<bfloat16_t> inputQkvUb_[2];
    LocalTensor<float> rawCosSinUb_[2];
    LocalTensor<int32_t> slotMappingUb_[2];
    LocalTensor<fp8_e4m3fn_t> vDataUb_;
    LocalTensor<fp8_e4m3fn_t> kDataUb_;
    LocalTensor<fp8_e8m0_t> kScaleUb_;
    LocalTensor<fp8_e4m3fn_t> qDataUb_;
    LocalTensor<fp8_e8m0_t> qScaleUb_;
    LocalTensor<uint8_t> mxScratchUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VEC_H_
