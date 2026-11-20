/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_INT8_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_INT8_H_

#include "kernel_operator.h"
#include "quant_flash_attn_tiling_data_int8.h"
#include "quant_flash_attn_common_int8.h"

namespace BaseApi {

using namespace AscendC;

// FIA-style INT8 MM1/MM2, FP16 scores/PV and external P compensation.
enum class Int8GqaCubeStrategy {
    SHARED_QP,
    M512,
    RESIDENT_Q
};

template <Int8GqaPath PATH = Int8GqaPath::BNSD, Int8GqaCubeStrategy STRATEGY = Int8GqaCubeStrategy::SHARED_QP,
          bool SCALE_FB = false, bool D64_K_REUSE = false, uint32_t STATIC_D = 0U>
class QuantFlashAttnBlockCubeGqaInt8 {
protected:
    static constexpr bool LARGE_M = STRATEGY == Int8GqaCubeStrategy::M512;
    static constexpr bool RESIDENT_Q = STRATEGY == Int8GqaCubeStrategy::RESIDENT_Q;
    static constexpr bool K_SCALE_PREFIX_CACHE = PATH == Int8GqaPath::TND && RESIDENT_Q;

public:
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* vDescale, __gm__ uint8_t* pScale, __gm__ uint8_t* workspaceCore,
                                const QuantFlashAttnTilingDataInt8& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        if constexpr (STATIC_D != 0U)
            tiling_.headDim = tiling_.headDimAlign = STATIC_D;
        if constexpr (PATH != Int8GqaPath::BNSD) {
            cacheK_ = QuantFlashAttnLayoutInt8<PATH>::QfaUseKScaleCache(tiling_);
        }
        if constexpr (K_SCALE_PREFIX_CACHE)
            cacheK_ = true;

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(query));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(value));
        vDescaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vDescale));
        GlobalTensor<float> pScaleGm;
        pScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pScale));
        pScaleValue_ = pScaleGm.GetValue(0);

        // Intermediate dynamic M tiles keep P and PV independent. Their two GM slots retain the non-cross-task
        // pipeline cadence.
        overlayPv_ = (tiling_.blockM == Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE ||
                      tiling_.blockM == Int8GqaTnd::QFA_LARGE_QUERY_TILE_ROWS);
        slotCount_ = tiling_.blockM >= Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE ? Int8GqaTnd::QFA_PV_OVERLAY_SLOTS :
                                                                                   Int8GqaTnd::QFA_PIPELINE_SLOTS;
        const uint32_t scoreBytes = slotCount_ * tiling_.scoreSlotElems * sizeof(half);
        const uint32_t pBytes = slotCount_ * tiling_.pSlotElems * sizeof(int8_t);
        scoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceCore));
        pGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspaceCore + scoreBytes));
        pvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspaceCore + scoreBytes + (overlayPv_ ? 0U : pBytes)));
        kDeqWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
            workspaceCore + tiling_.coreWorkspaceBytes -
            Int8GqaTnd::QfaKScaleWorkspaceElements(cacheK_, K_SCALE_PREFIX_CACHE, slotCount_, tiling_.blockN) *
                sizeof(uint64_t)));

        const uint32_t qL1Bytes = LARGE_M ? Align16(tiling_.blockM) * tiling_.headDimAlign : L1_Q_ELEMS;
        const uint32_t pL1Bytes = LARGE_M ? Align16(tiling_.blockM) * tiling_.blockN : L1_P_ELEMS;
        const uint32_t originalKvBytes =
            LARGE_M ? Int8GqaTnd::QFA_DEFAULT_KV_TILE_TOKENS * tiling_.headDimAlign : L1_KV_ELEMS;
        const uint32_t kScaleL1Bytes =
            (cacheK_ && !K_SCALE_PREFIX_CACHE ? Int8GqaTnd::QFA_K_SCALE_CACHE_TOKENS : tiling_.blockN) *
            sizeof(uint64_t);
        // Divide spare L1 between immutable K/V panels after reserving Q/P/scales.
        constexpr uint32_t L1_BYTES = 512U * 1024U;
        const uint32_t fixedL1Bytes = qL1Bytes + pL1Bytes + kScaleL1Bytes;
        // K and V split the remaining L1 space equally.
        const uint32_t perKvBudget = fixedL1Bytes < L1_BYTES ? (L1_BYTES - fixedL1Bytes) / 2U : 0U;
        residentPanelElems_ = tiling_.blockN * tiling_.headDimAlign;
        const uint32_t desiredPanels =
            tiling_.seqLenKv / tiling_.blockN + (tiling_.seqLenKv % tiling_.blockN != 0U ? 1U : 0U);
        residentPanels_ = Min32(MAX_RESIDENT_PANELS, Min32(desiredPanels, perKvBudget / residentPanelElems_));
        residentPanels_ = residentPanels_ == 0U ? 1U : residentPanels_;
        const uint32_t residentKvBytes = residentPanels_ * residentPanelElems_;
        const uint32_t kvL1Bytes = residentKvBytes > originalKvBytes ? residentKvBytes : originalKvBytes;
        pipe->InitBuffer(qL1Buf_, qL1Bytes);
        pipe->InitBuffer(kL1Buf_, kvL1Bytes);
        pipe->InitBuffer(vL1Buf_, kvL1Bytes);
        pipe->InitBuffer(pL1Buf_, pL1Bytes * sizeof(int8_t));
        pipe->InitBuffer(kDeqL1Buf_, kScaleL1Bytes);
        if constexpr (SCALE_FB) {
            pipe->InitBuffer(kDeqFbBuf_, K_DEQ_FB_ELEMS * sizeof(uint64_t));
        }

        // Shared Q/P uses two 32 KiB L0A panels; resident-Q uses two Q panels plus two P panels.
        pipe->InitBuffer(aL0Buf_,
                         (RESIDENT_Q ? L0_PING_PONG_SLOTS * 2U : L0_PING_PONG_SLOTS) * L0_AB_ELEMS * sizeof(int8_t));
        pipe->InitBuffer(bL0Buf_, L0_PING_PONG_SLOTS * L0_AB_ELEMS * sizeof(int8_t));
        pipe->InitBuffer(cL0Buf_, L0_PING_PONG_SLOTS * L0_C_ELEMS * sizeof(int32_t));

        qL1_ = qL1Buf_.Get<int8_t>();
        kL1_ = kL1Buf_.Get<int8_t>();
        vL1_ = vL1Buf_.Get<int8_t>();
        pL1_ = pL1Buf_.Get<int8_t>();
        kDeqL1_ = kDeqL1Buf_.Get<uint64_t>();
        if constexpr (SCALE_FB) {
            kDeqFb_ = kDeqFbBuf_.Get<uint64_t>();
        }
        aL0_ = aL0Buf_.Get<int8_t>();
        bL0_ = bL0Buf_.Get<int8_t>();
        cL0_ = cL0Buf_.Get<int32_t>();

        for (uint32_t i = 0; i < L0_PING_PONG_SLOTS; ++i) {
            SetFlag<HardEvent::M_MTE1>(AB_READY_EVENT + i);
            SetFlag<HardEvent::FIX_M>(C_READY_EVENT + i);
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

    // Only the current K producer selects the cache key. Rebinding an older PV task must not invalidate the next Q
    // task's cached scales.
    __aicore__ inline void SelectKTask(uint32_t batch, uint32_t kvHead)
    {
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

    // Rebind a staged task without reloading Q. Only the PV stage needs V descale.
    __aicore__ inline void SelectTask(uint32_t batch, uint32_t kvHead, uint32_t mStart, uint32_t mSize,
                                      bool selectVScale = true)
    {
        batch_ = batch;
        kvHead_ = kvHead;
        if (selectVScale && lastVHead_ != kvHead) {
            vDeqBits_ = PvFixpipeDeqBits(vDescaleGm_.GetValue(kvHead), pScaleValue_, tiling_.blockN);
            lastVHead_ = kvHead;
        }
        mSize_ = mSize;
        mAligned_ = Align16(mSize);
    }

    __aicore__ inline void BeginTask(uint32_t batch, uint32_t kvHead, uint32_t mStart, uint32_t mSize,
                                     bool selectVScale = true)
    {
        SelectKTask(batch, kvHead);
        SelectTask(batch, kvHead, mStart, mSize, selectVScale);
        QfaSyncEvent<HardEvent::MTE1_MTE2>(Q_EVENT);
        if (PATH == Int8GqaPath::BNSD || !QuantFlashAttnLayoutInt8<PATH>::IsSg(tiling_.layoutQ)) {
            const uint64_t offset =
                ((static_cast<uint64_t>(batch) * tiling_.numHeadsQ + kvHead * tiling_.groupSize) * tiling_.seqLenQ +
                 mStart) *
                tiling_.headDim;
            CopyNdToNz(qL1_, qGm_, offset, mSize, tiling_.headDim, tiling_.headDim, mAligned_);
        } else {
            QuantFlashAttnLayoutInt8<PATH>::QfaCopySgQuery(qL1_, qGm_, tiling_, batch, kvHead, mStart, mSize,
                                                           mAligned_);
        }

        QfaSyncEvent<HardEvent::MTE2_MTE1>(Q_EVENT);
        if constexpr (RESIDENT_Q) {
            // A new Q task may start before the preceding PV/Vec2. Protect resident L0A Q panels from the preceding
            // MM1's outstanding reads.
            QfaSyncEvent<HardEvent::M_MTE1>(Q_PANEL_LOADED_EVENT);
            // Keep the one or two Q (M,K) panels resident across every N tile. Host tiling guarantees ceil(M/128) *
            // ceil(D/128) <= 2.
            const uint32_t qKPanels = CeilDiv(tiling_.headDim, L0_BASE);
            for (uint32_t m0 = 0; m0 < mSize_; m0 += L0_BASE) {
                const uint32_t m0Size = Min32(L0_BASE, mSize_ - m0);
                for (uint32_t k0 = 0; k0 < tiling_.headDim; k0 += L0_BASE) {
                    const uint32_t k0Size = Min32(L0_BASE, tiling_.headDim - k0);
                    const uint32_t qSlot = (m0 / L0_BASE) * qKPanels + (k0 / L0_BASE);
                    LoadA(qSlot, qL1_, mAligned_, m0, m0Size, k0, k0Size);
                }
            }
            QfaSyncEvent<HardEvent::MTE1_M>(Q_PANEL_LOADED_EVENT);
        }
    }

    __aicore__ inline void ComputeMm1(uint32_t nStart, uint32_t nSize, uint32_t slot)
    {
        const uint32_t panelRows = nSize;
        const uint32_t nAligned = Align16(panelRows);
        const uint64_t kOffset = QfaRow(tiling_.layoutKv, batch_, kvHead_, nStart, tiling_.numHeadsKv, tiling_.seqLenKv,
                                        tiling_.kvTokenStart) *
                                 tiling_.headDim;
        // Cache immutable KV panels in disjoint L1 buffers, keyed by sequence base, length and panel index.
        const bool reuseK = SelectPanel<true>(nStart, kOffset);
        if (!reuseK) {
            QfaSyncEvent<HardEvent::MTE1_MTE2>(K_EVENT);
            if (panelRows != nAligned) {
                // ND2NZ's D padding does not imply zero-filled key rows. MMAD consumes an N tile aligned to 16 even
                // at the final KV tail.
                InitConstValueParams<uint16_t> zeros(Align32(tiling_.headDim) / INT8_C0, nAligned - panelRows,
                                                     panelRows, 0U);
                Fill(kL1_.template ReinterpretCast<uint16_t>()[panelRows *
                                                               (QFA_INT8_DMA_BLOCK_BYTES / sizeof(uint16_t))],
                     zeros);
                PipeBarrier<PIPE_MTE2>();
            }
            CopyNdToNz(kL1_, kGm_, kOffset, panelRows, tiling_.headDim,
                       tiling_.headDim * (tiling_.layoutKv == Int8GqaTnd::QFA_LAYOUT_BNSD ? 1U : tiling_.numHeadsKv),
                       nAligned);
            QfaSyncEvent<HardEvent::MTE2_MTE1>(K_EVENT);
        }

        PrepareKScale(panelRows, slot, nStart);
        if constexpr (SCALE_FB) {
            // Quant_PRE has 256 uint64_t entries. Refill this bounded window lazily; a new KV panel invalidates the
            // preceding panel's scales.
            kDeqFbTokenStart_ = Int8GqaTnd::QFA_INVALID_INDEX;
            kDeqFbNAligned_ = nAligned;
        }

        if constexpr (D64_K_REUSE && LARGE_M) {
            if (tiling_.headDim == Int8GqaTnd::QFA_HEAD_DIM_64 &&
                tiling_.blockM > Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE && mSize_ > LARGE_MM1_GROUP_ROWS) {
                ComputeMm1D64KReuse(nSize, nAligned, slot);
                return;
            }
        }

        if constexpr (STRATEGY == Int8GqaCubeStrategy::SHARED_QP) {
            if (tiling_.headDim > Int8GqaTnd::QFA_HEAD_DIM_128 && tiling_.headDim <= MM1_K_STEP &&
                tiling_.blockM < Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE && tiling_.seqLenKv <= tiling_.blockN) {
                // A resident single KV panel has no next-panel MTE1 traffic. Shared-QP already uses one full-D MMAD
                // per N128 tile. Preload the next Q/K panels in the other 32KiB AB slot while the current
                // MMAD/Fixpipe drains, without splitting K.
                ComputeMm1Large(nStart, nSize, nAligned, slot, 0U, mSize_);
                return;
            }
        }

        if constexpr (LARGE_M) {
            if (tiling_.blockM > Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE) {
                for (uint32_t mGroup = 0U; mGroup < mSize_; mGroup += LARGE_MM1_GROUP_ROWS) {
                    ComputeMm1Large(nStart, nSize, nAligned, slot, mGroup,
                                    Min32(LARGE_MM1_GROUP_ROWS, mSize_ - mGroup));
                }
                return;
            }
        }
        if (tiling_.blockM == Int8GqaTnd::QFA_PV_OVERLAY_MIN_QUERY_TILE) {
            // One B(K) panel serves both 128-row M subtiles. Preload the following N panel into the other L0B slot
            // while M/FIX run.
            ComputeMm1Large(nStart, nSize, nAligned, slot, 0U, mSize_);
            return;
        }

        for (uint32_t m0 = 0; m0 < mSize_; m0 += L0_BASE) {
            const uint32_t m0Size = Min32(L0_BASE, mSize_ - m0);
            for (uint32_t n0 = 0; n0 < nSize; n0 += L0_BASE) {
                const uint32_t n0Size = Min32(L0_BASE, nSize - n0);
                const uint32_t cSlot = cSlot_;
                WaitFlag<HardEvent::FIX_M>(C_READY_EVENT + cSlot);
                for (uint32_t k0 = 0; k0 < tiling_.headDim; k0 += MM1_K_STEP) {
                    const uint32_t k0Size = Min32(MM1_K_STEP, tiling_.headDim - k0);
                    const uint32_t abSlot = abSlot_;
                    WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + abSlot);
                    if constexpr (!RESIDENT_Q) {
                        LoadA(abSlot, qL1_, mAligned_, m0, m0Size, k0, k0Size);
                    }
                    LoadKTranspose(abSlot, nAligned, n0, n0Size, k0, k0Size);
                    QfaSyncEvent<HardEvent::MTE1_M>(AB_LOADED_EVENT);

                    const uint32_t aSlot =
                        RESIDENT_Q ? (m0 / L0_BASE) * CeilDiv(tiling_.headDim, L0_BASE) + k0 / L0_BASE : abSlot;
                    MmadTile(cSlot, aSlot, abSlot, Align16(m0Size), Align16(n0Size), k0Size, k0 == 0U);
                    if (k0 + k0Size < tiling_.headDim) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(AB_READY_EVENT + abSlot);
                    abSlot_ ^= 1U;
                }
                QfaSyncEvent<HardEvent::M_FIX>(C_READY_EVENT + cSlot);
                FixpipeScore(slot * tiling_.scoreSlotElems + m0 * tiling_.blockN + n0, tiling_.blockN, m0Size, n0Size,
                             n0, cSlot);
                SetFlag<HardEvent::FIX_M>(C_READY_EVENT + cSlot);
                cSlot_ ^= 1U;
            }
        }
    }

    __aicore__ inline void ComputeMm1Large(uint32_t nStart, uint32_t nSize, uint32_t nAligned, uint32_t slot,
                                           uint32_t mGroup, uint32_t groupSize)
    {
        // Large tasks are processed in groups of at most 256 rows, each with up to two 128-row MMAD subtiles and
        // one K chunk for D64/D128.
        uint32_t n0 = 0U;
        uint32_t n0Size = Min32(L0_BASE, nSize);
        uint32_t bSlot = abSlot_;
        WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + bSlot);
        LoadKTranspose(bSlot, nAligned, n0, n0Size, 0U, tiling_.headDim);
        if constexpr (!RESIDENT_Q) {
            LoadA(bSlot, qL1_, mAligned_, mGroup, Min32(L0_BASE, groupSize), 0U, tiling_.headDim);
        }
        SetFlag<HardEvent::MTE1_M>(AB_LOADED_EVENT);

        while (n0 < nSize) {
            WaitFlag<HardEvent::MTE1_M>(AB_LOADED_EVENT);

            const uint32_t c0 = cSlot_;
            WaitFlag<HardEvent::FIX_M>(C_READY_EVENT + c0);
            MmadTile(c0, RESIDENT_Q ? 0U : bSlot, bSlot, Align16(Min32(L0_BASE, groupSize)), Align16(n0Size),
                     tiling_.headDim);
            SetFlag<HardEvent::M_FIX>(C_READY_EVENT + c0);

            const bool hasSecondM = groupSize > L0_BASE;
            const uint32_t c1 = c0 ^ 1U;
            if (hasSecondM) {
                const uint32_t qSlot = RESIDENT_Q ? 1U : bSlot ^ 1U;
                if constexpr (!RESIDENT_Q) {
                    WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + qSlot);
                    LoadA(qSlot, qL1_, mAligned_, mGroup + L0_BASE, groupSize - L0_BASE, 0U, tiling_.headDim);
                    QfaSyncEvent<HardEvent::MTE1_M>(AB_LOADED_EVENT);
                }
                WaitFlag<HardEvent::FIX_M>(C_READY_EVENT + c1);
                MmadTile(c1, qSlot, bSlot, Align16(groupSize - L0_BASE), Align16(n0Size), tiling_.headDim);
                SetFlag<HardEvent::M_FIX>(C_READY_EVENT + c1);
                if constexpr (!RESIDENT_Q)
                    SetFlag<HardEvent::M_MTE1>(AB_READY_EVENT + qSlot);
            }

            // L0B remains live until both MMADs have read this K panel.
            SetFlag<HardEvent::M_MTE1>(AB_READY_EVENT + bSlot);
            abSlot_ ^= 1U;
            if (n0 + L0_BASE < nSize) {
                const uint32_t nextN = n0 + L0_BASE;
                const uint32_t nextNSize = Min32(L0_BASE, nSize - nextN);
                const uint32_t nextB = abSlot_;
                WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + nextB);
                LoadKTranspose(nextB, nAligned, nextN, nextNSize, 0U, tiling_.headDim);
                if constexpr (!RESIDENT_Q) {
                    LoadA(nextB, qL1_, mAligned_, mGroup, Min32(L0_BASE, groupSize), 0U, tiling_.headDim);
                }
                SetFlag<HardEvent::MTE1_M>(AB_LOADED_EVENT);
            }

            // Preload B on MTE1 while the two L0C slots drain independently.
            WaitFlag<HardEvent::M_FIX>(C_READY_EVENT + c0);
            FixpipeScore(slot * tiling_.scoreSlotElems + mGroup * tiling_.blockN + n0, tiling_.blockN,
                         Min32(L0_BASE, groupSize), n0Size, n0, c0);
            SetFlag<HardEvent::FIX_M>(C_READY_EVENT + c0);
            cSlot_ ^= 1U;
            if (hasSecondM) {
                WaitFlag<HardEvent::M_FIX>(C_READY_EVENT + c1);
                FixpipeScore(slot * tiling_.scoreSlotElems + (mGroup + L0_BASE) * tiling_.blockN + n0, tiling_.blockN,
                             groupSize - L0_BASE, n0Size, n0, c1);
                SetFlag<HardEvent::FIX_M>(C_READY_EVENT + c1);
                cSlot_ ^= 1U;
            }

            n0 += L0_BASE;
            if (n0 < nSize) {
                n0Size = Min32(L0_BASE, nSize - n0);
            }
            bSlot = abSlot_;
        }
    }

    // Preload immutable V while Vec1 runs; the preceding MM2 has retired its L1 readers.
    __aicore__ inline void PreloadV(uint32_t nStart, uint32_t nSize)
    {
        const uint32_t nAligned = Align32(nSize);
        const uint64_t vOffset = QfaRow(tiling_.layoutKv, batch_, kvHead_, nStart, tiling_.numHeadsKv, tiling_.seqLenKv,
                                        tiling_.kvTokenStart) *
                                 tiling_.headDim;
        const bool reuseV = SelectPanel<false>(nStart, vOffset);
        if (!reuseV) {
            QfaSyncEvent<HardEvent::MTE1_MTE2>(V_EVENT);
            if (nSize != nAligned) {
                // ND2NZ pads D to C0, but does not guarantee that unused KV rows are zero. L0B's INT8 transpose
                // reads complete groups of 32 rows.
                InitConstValueParams<uint16_t> zeros(Align32(tiling_.headDim) / INT8_C0, nAligned - nSize, nSize, 0U);
                Fill(vL1_.template ReinterpretCast<uint16_t>()[nSize * (QFA_INT8_DMA_BLOCK_BYTES / sizeof(uint16_t))],
                     zeros);
                PipeBarrier<PIPE_MTE2>();
            }
            CopyNdToNz(vL1_, vGm_, vOffset, nSize, tiling_.headDim,
                       tiling_.headDim * (tiling_.layoutKv == Int8GqaTnd::QFA_LAYOUT_BNSD ? 1U : tiling_.numHeadsKv),
                       nAligned);
            QfaSyncEvent<HardEvent::MTE2_MTE1>(V_EVENT);
        }
    }

    // Keep one K N128 panel live across M128 subtiles; AB_READY protects the ping-pong slots.
    __aicore__ inline void ComputeMm1D64KReuse(uint32_t nSize, uint32_t nAligned, uint32_t slot)
    {
        uint32_t bSlot = abSlot_;
        WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + bSlot);
        LoadKTranspose(bSlot, nAligned, 0U, Min32(L0_BASE, nSize), 0U, tiling_.headDim);
        LoadA(bSlot, qL1_, mAligned_, 0U, L0_BASE, 0U, tiling_.headDim);
        SetFlag<HardEvent::MTE1_M>(AB_LOADED_EVENT);

        for (uint32_t n0 = 0U; n0 < nSize; n0 += L0_BASE) {
            const uint32_t n0Size = Min32(L0_BASE, nSize - n0);
            const bool hasNextN = n0 + L0_BASE < nSize;
            for (uint32_t m0 = 0U; m0 < mSize_; m0 += L0_BASE) {
                const uint32_t aSlot = bSlot ^ ((m0 / L0_BASE) & 1U);
                const uint32_t c = cSlot_;
                WaitFlag<HardEvent::MTE1_M>(AB_LOADED_EVENT);
                WaitFlag<HardEvent::FIX_M>(C_READY_EVENT + c);
                MmadTile(c, aSlot, bSlot, Align16(Min32(L0_BASE, mSize_ - m0)), Align16(n0Size), tiling_.headDim);
                SetFlag<HardEvent::M_FIX>(C_READY_EVENT + c);
                SetFlag<HardEvent::M_MTE1>(AB_READY_EVENT + aSlot);

                if (m0 + L0_BASE < mSize_) {
                    const uint32_t nextM = m0 + L0_BASE;
                    const uint32_t nextA = aSlot ^ 1U;
                    WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + nextA);
                    LoadA(nextA, qL1_, mAligned_, nextM, Min32(L0_BASE, mSize_ - nextM), 0U, tiling_.headDim);
                    if (nextM == L0_BASE && hasNextN) {
                        // The old use of nextA's B slot ended in the preceding N panel. This copy overlaps current
                        // MMAD.
                        const uint32_t nextN = n0 + L0_BASE;
                        LoadKTranspose(bSlot ^ 1U, nAligned, nextN, Min32(L0_BASE, nSize - nextN), 0U, tiling_.headDim);
                    }
                    SetFlag<HardEvent::MTE1_M>(AB_LOADED_EVENT);
                } else if (hasNextN) {
                    // K for the next N panel was already preloaded. Reuse its B slot, but wait for its Q A slot
                    // before reloading.
                    const uint32_t nextB = bSlot ^ 1U;
                    WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + nextB);
                    LoadA(nextB, qL1_, mAligned_, 0U, L0_BASE, 0U, tiling_.headDim);
                    SetFlag<HardEvent::MTE1_M>(AB_LOADED_EVENT);
                }

                WaitFlag<HardEvent::M_FIX>(C_READY_EVENT + c);
                FixpipeScore(slot * tiling_.scoreSlotElems + m0 * tiling_.blockN + n0, tiling_.blockN,
                             Min32(L0_BASE, mSize_ - m0), n0Size, n0, c);
                SetFlag<HardEvent::FIX_M>(C_READY_EVENT + c);
                cSlot_ ^= 1U;
            }
            bSlot ^= 1U;
            abSlot_ ^= 1U;
        }
    }

    // Preload immutable V while Vec1 runs; the preceding MM2 has retired its L1 readers.
    __aicore__ inline void ComputeMm2(uint32_t nStart, uint32_t nSize, uint32_t slot, bool vPreloaded = false)
    {
        // The INT8 L1->L0B transpose uses the cached panel's full stride.
        const uint32_t nAligned = Align32(nSize);
        QfaSyncEvent<HardEvent::MTE1_MTE2>(P_EVENT);
        CopyNdToNz(pL1_, pGm_, slot * tiling_.pSlotElems, mSize_, nSize, tiling_.blockN, mAligned_);
        QfaSyncEvent<HardEvent::MTE2_MTE1>(P_EVENT);
        if (!vPreloaded) {
            PreloadV(nStart, nSize);
        }

        for (uint32_t m0 = 0; m0 < mSize_; m0 += L0_BASE) {
            const uint32_t m0Size = Min32(L0_BASE, mSize_ - m0);
            for (uint32_t d0 = 0; d0 < tiling_.headDim; d0 += L0_BASE) {
                const uint32_t d0Size = Min32(L0_BASE, tiling_.headDim - d0);
                const uint32_t cSlot = cSlot_;
                WaitFlag<HardEvent::FIX_M>(C_READY_EVENT + cSlot);
                for (uint32_t k0 = 0; k0 < nSize; k0 += MM2_K_STEP) {
                    const uint32_t k0Size = Min32(MM2_K_STEP, nSize - k0);
                    const uint32_t abSlot = abSlot_;
                    WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + abSlot);
                    LoadA((RESIDENT_Q ? L0_PING_PONG_SLOTS : 0U) + abSlot, pL1_, mAligned_, m0, m0Size, k0, k0Size);
                    LoadVNormal(abSlot, nAligned, d0, d0Size, k0, k0Size);
                    QfaSyncEvent<HardEvent::MTE1_M>(AB_LOADED_EVENT);

                    // INT8 transpose requires a physical N extent aligned to 32 (including D72).
                    MmadTile(cSlot, (RESIDENT_Q ? L0_PING_PONG_SLOTS : 0U) + abSlot, abSlot, Align16(m0Size),
                             Align32(d0Size), k0Size, k0 == 0U);
                    if (k0 + k0Size < nSize) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(AB_READY_EVENT + abSlot);
                    abSlot_ ^= 1U;
                }
                QfaSyncEvent<HardEvent::M_FIX>(C_READY_EVENT + cSlot);
                // Each 128-row P region also holds its PV tile after MM2 has finished reading P into L1.
                const uint32_t pvOffset =
                    overlayPv_ ? slot * tiling_.pSlotElems / sizeof(half) + m0 * (tiling_.blockN / sizeof(half)) + d0 :
                                 slot * tiling_.pvSlotElems + m0 * tiling_.headDimAlign + d0;
                FixpipeHalf(pvGm_, pvOffset, tiling_.headDimAlign, m0Size, d0Size, cSlot, vDeqBits_);
                SetFlag<HardEvent::FIX_M>(C_READY_EVENT + cSlot);
                cSlot_ ^= 1U;
            }
        }
    }

    __aicore__ inline void Finish()
    {
        for (uint32_t i = 0; i < L0_PING_PONG_SLOTS; ++i) {
            WaitFlag<HardEvent::M_MTE1>(AB_READY_EVENT + i);
            WaitFlag<HardEvent::FIX_M>(C_READY_EVENT + i);
        }
    }

protected:
    static constexpr uint32_t MAX_RESIDENT_PANELS = 16U;
    // Pin the early panels; use the final slot as a streaming fallback when the full sequence exceeds capacity.
    // This retains reuse even for long scans whose working set cannot fit in L1.
    __aicore__ inline uint32_t ResidentPanelSlot(uint32_t nStart) const
    {
        return Min32(nStart / tiling_.blockN, residentPanels_ - 1U);
    }

    // nSize is fixed by (sequence length, nStart). The sequence's exact GM base and length therefore validate all
    // pinned panel tails together. Only the final streaming slot needs a separate N identity.
    __aicore__ inline bool SelectResidentPanel(uint32_t nStart, uint64_t offset, uint64_t& lastBase,
                                               uint32_t& lastLength, uint32_t& valid, uint32_t& streamStart)
    {
        const uint64_t tokenStride = static_cast<uint64_t>(tiling_.headDim) *
                                     (tiling_.layoutKv == Int8GqaTnd::QFA_LAYOUT_BNSD ? 1U : tiling_.numHeadsKv);
        const uint64_t base = offset - static_cast<uint64_t>(nStart) * tokenStride;
        if (lastBase != base || lastLength != tiling_.seqLenKv) {
            lastBase = base;
            lastLength = tiling_.seqLenKv;
            valid = 0U;
        }
        const uint32_t slot = ResidentPanelSlot(nStart);
        const bool streaming = slot + 1U == residentPanels_;
        const bool hit = (valid & (1U << slot)) != 0U && (!streaming || streamStart == nStart);
        valid |= 1U << slot;
        if (streaming)
            streamStart = nStart;
        return hit;
    }

    template <bool KEY>
    __aicore__ inline bool SelectPanel(uint32_t nStart, uint64_t offset)
    {
        auto& local = KEY ? kL1_ : vL1_;
        auto& buffer = KEY ? kL1Buf_ : vL1Buf_;
        if constexpr (STRATEGY != Int8GqaCubeStrategy::SHARED_QP) {
            if (residentPanels_ == 1U && tiling_.seqLenKv > tiling_.blockN) {
                local = buffer.Get<int8_t>();
                return false;
            }
        }
        local = buffer.Get<int8_t>()[ResidentPanelSlot(nStart) * residentPanelElems_];
        return SelectResidentPanel(nStart, offset, KEY ? residentKBase_ : residentVBase_,
                                   KEY ? residentKSeqLen_ : residentVSeqLen_, KEY ? residentKValid_ : residentVValid_,
                                   KEY ? residentKStreamStart_ : residentVStreamStart_);
    }

    static constexpr uint32_t L0_BASE = 128;
    static constexpr uint32_t LARGE_MM1_GROUP_ROWS = 256U;
    static constexpr uint32_t L0_PING_PONG_SLOTS = 2U;
    static constexpr uint32_t CUBE_MN_ALIGNMENT = 16U;
    static constexpr uint32_t INT8_C0 = QFA_INT8_DMA_BLOCK_BYTES / sizeof(int8_t);
    static constexpr uint32_t FIXPIPE_L1_TO_FB_BLOCK_BYTES = 128U;
    // The M512 path uses D64 or D128 and streams 256-row M groups through two 32 KiB L0A/L0B panels.
    static constexpr uint32_t L1_Q_ELEMS = Int8GqaTnd::QFA_Q_L1_TILE_BYTES / sizeof(int8_t);
    static constexpr uint32_t L1_KV_ELEMS = 512 * 256;
    static constexpr uint32_t L1_P_ELEMS = (LARGE_M ? 512U : 256U) * 512U;
    static constexpr uint32_t L0_AB_ELEMS = 128U * (RESIDENT_Q ? 128U : 256U);
    static constexpr uint32_t MM2_K_STEP = RESIDENT_Q ? 128U : 256U;
    static constexpr uint32_t MM1_K_STEP = RESIDENT_Q ? 128U : 256U;
    static constexpr uint32_t L0_C_ELEMS = 128 * 128;
    static constexpr uint32_t K_DEQ_FB_ELEMS = 256U;
    static constexpr uint32_t Q_EVENT = EVENT_ID0;
    static constexpr uint32_t K_EVENT = EVENT_ID1;
    static constexpr uint32_t V_EVENT = EVENT_ID2;
    static constexpr uint32_t P_EVENT = EVENT_ID3;
    static constexpr uint32_t AB_READY_EVENT = EVENT_ID3;
    static constexpr uint32_t C_READY_EVENT = EVENT_ID3;
    static constexpr uint32_t Q_PANEL_LOADED_EVENT = EVENT_ID6;
    static constexpr uint32_t AB_LOADED_EVENT = EVENT_ID7;

    __aicore__ inline static uint32_t Align16(uint32_t x)
    {
        return (x + CUBE_MN_ALIGNMENT - 1U) & ~(CUBE_MN_ALIGNMENT - 1U);
    }
    __aicore__ inline static uint32_t Align32(uint32_t x)
    {
        return (x + INT8_C0 - 1U) & ~(INT8_C0 - 1U);
    }
    __aicore__ inline static uint32_t CeilDiv(uint32_t x, uint32_t y)
    {
        return (x + y - 1U) / y;
    }
    __aicore__ inline static uint32_t Min32(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }

    __aicore__ inline void MmadTile(uint32_t c, uint32_t a, uint32_t b, uint32_t m, uint32_t n, uint32_t k,
                                    bool initialize = true)
    {
        MmadParams params(m, n, k, 0U, false, initialize);
        Mmad(cL0_[c * L0_C_ELEMS], aL0_[a * L0_AB_ELEMS], bL0_[b * L0_AB_ELEMS], params);
    }

    __aicore__ inline void CopyNdToNz(const LocalTensor<int8_t>& dst, const GlobalTensor<int8_t>& src,
                                      uint64_t srcOffset, uint32_t rows, uint32_t cols, uint32_t srcStride,
                                      uint32_t alignedRows)
    {
        Nd2NzParams p(1U, rows, cols, 0U, srcStride, alignedRows, 1U, 0U);
        if constexpr (PATH == Int8GqaPath::BNSD) {
            DataCopy(dst, src[srcOffset], p);
        } else {
            if (srcStride > UINT16_MAX) {
                // Nd2Nz srcDValue is uint16 on A2/A3. Match FIA's row fallback.
                p.nValue = 1U;
                p.srcDValue = cols;
                for (uint32_t row = 0; row < rows; ++row) {
                    DataCopy(dst[row * INT8_C0], src[srcOffset + static_cast<uint64_t>(row) * srcStride], p);
                }
            } else {
                DataCopy(dst, src[srcOffset], p);
            }
        }
    }

    __aicore__ inline void LoadA(uint32_t abSlot, const LocalTensor<int8_t>& src, uint32_t mL1Aligned, uint32_t m0,
                                 uint32_t mSize, uint32_t k0, uint32_t kSize)
    {
        // A2 INT8 LoadData3D needs Win=16; Win=32 drops half the K extent.
        constexpr uint32_t loadWindow = 16U;
        constexpr uint8_t maxLoadDataPadding = 255U;
        const uint8_t padList[] = {0, 0, 0, maxLoadDataPadding};
        LoadData3DParamsV2<int8_t> p(padList, mL1Aligned / CUBE_MN_ALIGNMENT, loadWindow, Align32(kSize),
                                     Align32(kSize), Align16(mSize), 0U, m0, 1U, 1U, 1U, 1U, 1U, 1U, false, false, 0);
        SetFmatrix(mL1Aligned / CUBE_MN_ALIGNMENT, loadWindow, padList, FmatrixMode::FMATRIX_LEFT);
        SetLoadDataPaddingValue<int8_t>(0);
        // Keep the feature-map base fixed; mStartPt selects rows within that feature map.
        const uint32_t srcOff = k0 * mL1Aligned;
        LoadData(aL0_[abSlot * L0_AB_ELEMS], src[srcOff], p);
    }

    __aicore__ inline void LoadKTranspose(uint32_t abSlot, uint32_t nAligned, uint32_t n0, uint32_t nSize, uint32_t k0,
                                          uint32_t kSize)
    {
        LoadData2DParams p(0U, CeilDiv(nSize, CUBE_MN_ALIGNMENT), 1U, 0U, 0U, false, 0U);
        const uint32_t kSlices = CeilDiv(kSize, INT8_C0);
        for (uint32_t ki = 0; ki < kSlices; ++ki) {
            const uint32_t srcOff = (k0 / INT8_C0 + ki) * nAligned * INT8_C0 + n0 * INT8_C0;
            const uint32_t dstOff = abSlot * L0_AB_ELEMS + ki * Align16(nSize) * INT8_C0;
            LoadData(bL0_[dstOff], kL1_[srcOff], p);
        }
    }

    __aicore__ inline void LoadVNormal(uint32_t abSlot, uint32_t kAligned, uint32_t d0, uint32_t dSize, uint32_t k0,
                                       uint32_t kSize)
    {
        // Unlike FP16 LoadData(ifTranspose=true), A2 INT8 B requires the dedicated 2D transpose path (as in IFA
        // full-quant MM2).
        LoadData2dTransposeParams p(0U, CeilDiv(dSize, INT8_C0), kAligned / INT8_C0, 1U, 0U);
        const uint32_t srcBase = d0 * kAligned + k0 * INT8_C0;
        const uint32_t dstBase = abSlot * L0_AB_ELEMS;
        for (uint32_t ki = 0; ki < CeilDiv(kSize, INT8_C0); ++ki) {
            const uint32_t srcOff = srcBase + ki * INT8_C0 * INT8_C0;
            const uint32_t dstOff = dstBase + ki * INT8_C0 * Align32(dSize);
            LoadDataWithTranspose(bL0_[dstOff], vL1_[srcOff], p);
        }
    }

    __aicore__ inline void PrepareKScale(uint32_t nSize, uint32_t slot, uint32_t nStart)
    {
        if constexpr (PATH == Int8GqaPath::BNSD)
            SelectKTask(batch_, kvHead_);
        kScaleBase_ = cacheK_ && !K_SCALE_PREFIX_CACHE ? nStart : 0U;
        if (!K_SCALE_PREFIX_CACHE && !NeedKScale(nStart))
            return;

        // Both paired AIV subcores encode disjoint halves of this tile in GM. The AIC waits for their MTE3
        // completion before calling this method.
        QfaSyncEvent<HardEvent::FIX_S>(EVENT_ID6);
        const uint32_t gmOffset =
            Int8GqaTnd::QfaKScaleGmOffset(cacheK_, K_SCALE_PREFIX_CACHE, nStart, slot, tiling_.blockN);
        DataCopy(kDeqL1_[kScaleBase_], kDeqWorkspaceGm_[gmOffset], Align16(nSize));

        QfaSyncEvent<HardEvent::MTE2_FIX>(EVENT_ID6);
        if constexpr (PATH != Int8GqaPath::BNSD) {
            if (CachedKScales(nStart))
                kCacheValid_ |= 1U << (nStart / tiling_.blockN);
        }
    }

    __aicore__ inline void FixpipeScore(uint32_t dstOffset, uint32_t dstStride, uint32_t mSize, uint32_t nSize,
                                        uint32_t tokenOffset, uint32_t cSlot)
    {
        FixpipeParamsV220 p(nSize, mSize, Align16(mSize), dstStride, false);
        p.ndNum = 1;
        p.quantPre = QuantMode_t::VDEQF16;
        if constexpr (SCALE_FB) {
            const uint32_t fbTokenStart = tokenOffset / K_DEQ_FB_ELEMS * K_DEQ_FB_ELEMS;
            if (kDeqFbTokenStart_ != fbTokenStart) {
                const uint32_t copyElems = Min32(K_DEQ_FB_ELEMS, kDeqFbNAligned_ - fbTokenStart);
                // Retire the preceding window's FIX readers before the overwrite, then make the new scale table
                // visible to FIX.
                PipeBarrier<PIPE_FIX>();
                DataCopyParams scaleCopy(
                    1U, static_cast<uint16_t>(copyElems * sizeof(uint64_t) / FIXPIPE_L1_TO_FB_BLOCK_BYTES), 0U, 0U);
                DataCopy(kDeqFb_, kDeqL1_[kScaleBase_ + fbTokenStart], scaleCopy);
                PipeBarrier<PIPE_FIX>();
                kDeqFbTokenStart_ = fbTokenStart;
            }
            SetFixPipeConfig<uint64_t, false>(kDeqFb_[tokenOffset - fbTokenStart], false);
            Fixpipe<half, int32_t, CFG_ROW_MAJOR>(scoreGm_[dstOffset], cL0_[cSlot * L0_C_ELEMS], p);
        } else {
            LocalTensor<uint64_t> scaleL1 = kDeqL1_[kScaleBase_ + tokenOffset];
            Fixpipe<half, int32_t, CFG_ROW_MAJOR>(scoreGm_[dstOffset], cL0_[cSlot * L0_C_ELEMS], scaleL1, p);
        }
    }

    __aicore__ inline void FixpipeHalf(GlobalTensor<half>& dst, uint32_t dstOffset, uint32_t dstStride, uint32_t mSize,
                                       uint32_t nSize, uint32_t cSlot, uint64_t deqBits)
    {
        FixpipeParamsV220 p(nSize, mSize, Align16(mSize), dstStride, false);
        p.ndNum = 1;
        p.quantPre = QuantMode_t::DEQF16;
        p.deqScalar = deqBits;
        Fixpipe<half, int32_t, CFG_ROW_MAJOR>(dst[dstOffset], cL0_[cSlot * L0_C_ELEMS], p);
    }

    QuantFlashAttnTilingDataInt8 tiling_{};
    uint32_t batch_ = 0, kvHead_ = 0, mSize_ = 0, mAligned_ = 0, abSlot_ = 0, cSlot_ = 0;
    uint32_t slotCount_ = Int8GqaTnd::QFA_PIPELINE_SLOTS;
    bool overlayPv_ = false;
    uint64_t vDeqBits_ = 0;
    bool cacheK_ = false;
    uint32_t residentPanelElems_ = 0U;
    uint32_t residentPanels_ = 1U;
    uint64_t residentKBase_ = ~uint64_t{0}, residentVBase_ = ~uint64_t{0};
    uint32_t residentKSeqLen_ = 0U, residentVSeqLen_ = 0U;
    uint32_t residentKValid_ = 0U, residentVValid_ = 0U;
    uint32_t residentKStreamStart_ = 0U, residentVStreamStart_ = 0U;
    uint32_t kCacheValid_ = 0U;
    uint32_t lastKBatch_ = Int8GqaTnd::QFA_INVALID_INDEX, lastKHead_ = Int8GqaTnd::QFA_INVALID_INDEX;
    uint32_t kScaleBase_ = 0U;
    uint32_t kDeqFbTokenStart_ = Int8GqaTnd::QFA_INVALID_INDEX;
    uint32_t kDeqFbNAligned_ = 0U;
    float pScaleValue_ = 0.0f;
    uint32_t lastVHead_ = Int8GqaTnd::QFA_INVALID_INDEX;

    GlobalTensor<int8_t> qGm_, kGm_, vGm_;
    GlobalTensor<uint64_t> kDeqWorkspaceGm_;
    TBuf<TPosition::A1> kDeqL1Buf_;
    LocalTensor<uint64_t> kDeqL1_;
    TBuf<TPosition::C2PIPE2GM> kDeqFbBuf_;
    LocalTensor<uint64_t> kDeqFb_;
    GlobalTensor<float> vDescaleGm_;
    GlobalTensor<half> scoreGm_;
    GlobalTensor<int8_t> pGm_;
    GlobalTensor<half> pvGm_;
    TBuf<TPosition::A1> qL1Buf_, kL1Buf_, vL1Buf_, pL1Buf_;
    TBuf<TPosition::A2> aL0Buf_;
    TBuf<TPosition::B2> bL0Buf_;
    TBuf<TPosition::CO1> cL0Buf_;
    LocalTensor<int8_t> qL1_, kL1_, vL1_, pL1_, aL0_, bL0_;
    LocalTensor<int32_t> cL0_;
};

} // namespace BaseApi
#endif
