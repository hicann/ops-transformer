/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BLOCK_EPILOGUE_ONLINE_SOFTMAX_FULL_QUANT_HPP
#define BLOCK_EPILOGUE_ONLINE_SOFTMAX_FULL_QUANT_HPP

#include "block_epilogue_full_quant.hpp"

namespace GbsaKernelArch22::Int8 {

template <class ElementO, bool singleRowTile, bool compactUb>
struct GbsaFullQuantVectorArch22<ElementO, singleRowTile, compactUb>::EpilogueOnlineSoftmax {
    static constexpr uint32_t ROW_ELEMENTS = BATCH_ROWS * FullQuantLayout::KV_TILE;
    static constexpr uint32_t Q_BROADCAST = BufferLayout::END;
    static_assert(Q_BROADCAST + FullQuantLayout::MAX_VECTOR_ROWS * 32U <= 192U * FullQuantLayout::KIB);

    GbsaFullQuantVectorArch22& state;

    static constexpr uint32_t K_CACHE_BASE = 64U * FullQuantLayout::KIB;
    static constexpr uint32_t K_CACHE_ELEMENTS = 128U * FullQuantLayout::KIB / sizeof(float);
    static constexpr uint32_t CACHE_REPEAT_MAP = CompactUbLayout::END + FullQuantLayout::VECTOR_ROWS * 32U;
    static constexpr uint32_t CACHE_INNER_OFFSETS = CACHE_REPEAT_MAP + FullQuantLayout::KV_TILE * sizeof(uint32_t);
    static constexpr uint32_t CACHE_INPUT_INDICES = CACHE_INNER_OFFSETS + 128U * sizeof(uint32_t);
    static constexpr uint32_t CACHE_GATHER_OFFSETS = CACHE_INPUT_INDICES + 128U * sizeof(uint32_t);
    static_assert(CACHE_GATHER_OFFSETS + FullQuantLayout::KV_TILE * sizeof(uint32_t) <= K_CACHE_BASE);

    template <bool isPaged, class Source>
    __aicore__ inline bool PrepareHeadKScaleCache(AscendC::GlobalTensor<float> scales, const Source& source,
                                                  uint32_t batch, uint32_t heads, uint32_t head, uint32_t kvLength,
                                                  uint64_t kvStart, uint64_t totalKvTokens, uint32_t blockY,
                                                  bool willReuse)
    {
        if constexpr (!compactUb) {
            return false;
        } else {
            // Exact spans already fill a 32-byte DMA block at Y>=8; keep their original direct scale path.
            if (blockY * sizeof(float) >= 32U || !willReuse || state.totalRows_ == 0U || kvLength == 0U ||
                kvLength > K_CACHE_ELEMENTS) {
                return false;
            }
            if (!state.cacheMapsReady_) {
                auto repeat = state.template Ub<uint32_t>(CACHE_REPEAT_MAP);
                auto inner = state.template Ub<uint32_t>(CACHE_INNER_OFFSETS);
                for (uint32_t i = 0; i < FullQuantLayout::KV_TILE; ++i) {
                    repeat.SetValue(i, (i / blockY) * sizeof(uint32_t));
                }
                for (uint32_t i = 0; i < FullQuantLayout::PAGE_TOKENS; ++i) {
                    inner.SetValue(i, (i % blockY) * sizeof(float));
                }
                state.template Sync<AscendC::HardEvent::S_V>();
                state.cacheMapsReady_ = true;
            }
            if (state.cachedKBatch_ == batch && state.cachedKHead_ == head && state.cachedKLength_ == kvLength) {
                return true;
            }
            auto cache = state.template Ub<float>(K_CACHE_BASE);
            state.template Sync<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
            if constexpr (isPaged) {
                for (uint32_t page = 0; page * FullQuantLayout::PAGE_TOKENS < kvLength; ++page) {
                    const uint32_t remain = kvLength - page * FullQuantLayout::PAGE_TOKENS;
                    const uint32_t length =
                        remain < FullQuantLayout::PAGE_TOKENS ? remain : FullQuantLayout::PAGE_TOKENS;
                    const uint64_t src = (static_cast<uint64_t>(source.PhysicalPage(page)) * heads + head) *
                                         FullQuantLayout::PAGE_TOKENS;
                    AscendC::DataCopyExtParams copy{1, length * uint32_t{sizeof(float)}, 0, 0, 0};
                    AscendC::DataCopyPad(cache[page * FullQuantLayout::PAGE_TOKENS], scales[src], copy, padding);
                }
            } else {
                const uint64_t src = static_cast<uint64_t>(head) * totalKvTokens + kvStart;
                AscendC::DataCopyExtParams copy{1, kvLength * uint32_t{sizeof(float)}, 0, 0, 0};
                AscendC::DataCopyPad(cache, scales[src], copy, padding);
            }
            state.template Sync<AscendC::HardEvent::MTE2_V>();
            state.cachedKBatch_ = batch;
            state.cachedKHead_ = head;
            state.cachedKLength_ = kvLength;
            return true;
        }
    }

    template <class Source>
    __aicore__ inline Tile PrefetchCachedKScale(const Source& source, uint32_t& cursor, uint32_t count, uint32_t round,
                                                uint32_t blockY)
    {
        Tile tile = source.DescribeTile(cursor, count);
        auto dst = state.template Ub<float>(BufferLayout::K_SCALE + (round % FullQuantLayout::DOUBLE_BUFFER) *
                                                                        FullQuantLayout::KV_TILE * sizeof(float));
        auto indices = state.template Ub<int32_t>(CACHE_INPUT_INDICES);
        auto offsets = state.template Ub<uint32_t>(CACHE_GATHER_OFFSETS);
        const uint32_t begin = tile.pages[0].firstIndex;
        const uint32_t end = tile.pages[FullQuantLayout::PAGES_PER_TILE - 1U].endIndex;
        const uint32_t explicitEnd = end < source.ExplicitCount() ? end : source.ExplicitCount();
        const uint32_t explicitBegin = begin < explicitEnd ? begin : explicitEnd;
        const uint32_t numIndices = explicitEnd - explicitBegin;
        AscendC::Duplicate(dst, 0.0f, FullQuantLayout::KV_TILE);
        state.template Sync<AscendC::HardEvent::V_MTE2>();
        if (numIndices != 0U) {
            AscendC::DataCopyExtParams copy{1, numIndices * uint32_t{sizeof(int32_t)}, 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> padding{false, 0, 0, 0};
            AscendC::DataCopyPad(indices, source.GetIndexTensor()[explicitBegin], copy, padding);
        }
        if (end > source.ExplicitCount()) {
            // The virtual residual has no explicit input index; append its logical block after DMA padding completes.
            state.template Sync<AscendC::HardEvent::MTE2_S>();
            indices.SetValue(numIndices, static_cast<int32_t>(source.ResidualId()));
            state.template Sync<AscendC::HardEvent::S_V>();
        } else {
            state.template Sync<AscendC::HardEvent::MTE2_V>();
        }
        const uint32_t offsetShift = static_cast<uint32_t>(__builtin_ctz(blockY)) + 2U;
        const bool fullTile =
            tile.pages[FullQuantLayout::PAGES_PER_TILE - 1U].validTokens == FullQuantLayout::PAGE_TOKENS;
        if (fullTile) {
            AscendC::Gather(offsets, indices.template ReinterpretCast<uint32_t>(),
                            state.template Ub<uint32_t>(CACHE_REPEAT_MAP), uint32_t{0}, FullQuantLayout::KV_TILE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(offsets, offsets, offsetShift, FullQuantLayout::KV_TILE);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t chunk = 0; chunk < FullQuantLayout::PAGES_PER_TILE; ++chunk) {
                auto part = offsets[chunk * FullQuantLayout::PAGE_TOKENS].template ReinterpretCast<int32_t>();
                AscendC::Add(part, part, state.template Ub<int32_t>(CACHE_INNER_OFFSETS), FullQuantLayout::PAGE_TOKENS);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(dst, state.template Ub<float>(K_CACHE_BASE), offsets, uint32_t{0},
                            FullQuantLayout::KV_TILE);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t chunk = 0; chunk < FullQuantLayout::PAGES_PER_TILE; ++chunk) {
                const auto& part = tile.pages[chunk];
                if (part.validTokens == 0U) {
                    continue;
                }
                const uint32_t countTokens = part.validTokens;
                auto chunkOffsets = offsets[chunk * FullQuantLayout::PAGE_TOKENS];
                AscendC::Gather(chunkOffsets, indices[part.firstIndex - begin].template ReinterpretCast<uint32_t>(),
                                state.template Ub<uint32_t>(CACHE_REPEAT_MAP), uint32_t{0}, countTokens);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftLeft(chunkOffsets, chunkOffsets, offsetShift, countTokens);
                AscendC::PipeBarrier<PIPE_V>();
                auto signedOffsets = chunkOffsets.template ReinterpretCast<int32_t>();
                AscendC::Add(signedOffsets, signedOffsets, state.template Ub<int32_t>(CACHE_INNER_OFFSETS),
                             countTokens);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(dst[chunk * FullQuantLayout::PAGE_TOKENS], state.template Ub<float>(K_CACHE_BASE),
                                chunkOffsets, uint32_t{0}, countTokens);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        return tile;
    }

    template <bool isPaged, class PlanType>
    __aicore__ inline void PrefetchKScale(AscendC::GlobalTensor<float> scales, const Tile& tile, uint32_t heads,
                                          uint32_t head, uint32_t round, uint64_t kvStart, uint64_t totalKvTokens,
                                          uint32_t blockY, const PlanType& plan)
    {
        if (state.totalRows_ == 0U) {
            return;
        }
        if (blockY >= 4U) {
            PrefetchExactKScale<isPaged>(scales, tile, heads, head, round, kvStart, totalKvTokens, blockY, plan);
            return;
        }
        const uint32_t slot = round % FullQuantLayout::DOUBLE_BUFFER;
        auto dst = state.template Ub<float>(BufferLayout::K_SCALE + slot * FullQuantLayout::KV_TILE * sizeof(float));
        constexpr uint32_t scratchFloats = BufferLayout::K_SCALE_CHUNK_BYTES / sizeof(float);
        // At most PAGE_TOKENS/blockY sparse blocks occur in one chunk. Give
        // each possible source window its share of scratch, bounded by a page
        // so a PA copy never crosses from this head into another head.
        const uint32_t perWindow = blockY * (scratchFloats / FullQuantLayout::PAGE_TOKENS);
        const uint32_t windowTokens =
            perWindow < FullQuantLayout::PAGE_TOKENS ? perWindow : FullQuantLayout::PAGE_TOKENS;
        const uint64_t windowMask = windowTokens - 1U;
        AscendC::Duplicate(dst, 0.0f, FullQuantLayout::KV_TILE);
        AscendC::Duplicate(state.template Ub<uint32_t>(BufferLayout::K_SCALE_OFFSETS), uint32_t{0},
                           FullQuantLayout::KV_TILE);
        state.template Sync<AscendC::HardEvent::V_MTE2>();
        state.template Sync<AscendC::HardEvent::V_S>();
        bool needsGather = false;
        for (uint32_t chunk = 0; chunk < FullQuantLayout::PAGES_PER_TILE; ++chunk) {
            const Page& part = tile.pages[chunk];
            if (part.validTokens == 0) {
                continue;
            }
            if (part.contiguous) {
                const uint64_t offset = ScaleOffset<isPaged>(part.sourceToken, heads, head, kvStart, totalKvTokens);
                AscendC::DataCopyExtParams copy{1, part.validTokens * uint32_t{sizeof(float)}, 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
                AscendC::DataCopyPad(dst[chunk * FullQuantLayout::PAGE_TOKENS], scales[offset], copy, padding);
                continue;
            }
            needsGather = true;
            auto scratch =
                state.template Ub<float>(BufferLayout::K_SCALE_GATHER + chunk * BufferLayout::K_SCALE_CHUNK_BYTES);
            auto offsets = state.template Ub<uint32_t>(BufferLayout::K_SCALE_OFFSETS +
                                                       chunk * FullQuantLayout::PAGE_TOKENS * sizeof(uint32_t));
            auto offsetsPtr = reinterpret_cast<__ubuf__ uint32_t*>(offsets.GetPhyAddr());
            uint64_t cachedWindow = UINT64_MAX;
            uint32_t windowSlot = 0;
            uint32_t windows = 0;
            uint32_t packed = 0;
            for (uint32_t ri = part.planBegin; ri < part.planEnd; ++ri) {
                const auto run = plan.GetSpan(ri);
                for (uint32_t i = 0; i < run.length;) {
                    const uint64_t token = run.token + i;
                    const uint64_t window = token & ~windowMask;
                    const uint32_t inner = token & windowMask;
                    if (window != cachedWindow) {
                        cachedWindow = window;
                        windowSlot = windows++ * windowTokens;
                        const uint64_t offset = ScaleOffset<isPaged>(window, heads, head, kvStart, totalKvTokens);
                        uint32_t length = windowTokens;
                        if constexpr (!isPaged) {
                            const uint64_t available = totalKvTokens - kvStart - window;
                            length = available < length ? available : length;
                        }
                        AscendC::DataCopyExtParams copy{1, length * uint32_t{sizeof(float)}, 0, 0, 0};
                        AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
                        AscendC::DataCopyPad(scratch[windowSlot], scales[offset], copy, padding);
                    }
                    const uint32_t remain = run.length - i;
                    const uint32_t take = remain < windowTokens - inner ? remain : windowTokens - inner;
                    for (uint32_t j = 0; j < take; ++j) {
                        offsetsPtr[packed++] = (windowSlot + inner + j) * sizeof(float);
                    }
                    i += take;
                }
            }
        }
        if (!needsGather) {
            return;
        }
        state.template Sync<AscendC::HardEvent::S_V>();
        state.template Sync<AscendC::HardEvent::MTE2_V>();
        for (uint32_t chunk = 0; chunk < FullQuantLayout::PAGES_PER_TILE; ++chunk) {
            const auto& part = tile.pages[chunk];
            if (part.validTokens != 0U && !part.contiguous) {
                auto scratch =
                    state.template Ub<float>(BufferLayout::K_SCALE_GATHER + chunk * BufferLayout::K_SCALE_CHUNK_BYTES);
                auto offsets = state.template Ub<uint32_t>(BufferLayout::K_SCALE_OFFSETS +
                                                           chunk * FullQuantLayout::PAGE_TOKENS * sizeof(uint32_t));
                AscendC::Gather(dst[chunk * FullQuantLayout::PAGE_TOKENS], scratch, offsets, uint32_t{0},
                                FullQuantLayout::PAGE_TOKENS);
            }
        }
        state.template Sync<AscendC::HardEvent::V_MTE2>();
    }

    __aicore__ inline void PrepareQScale(float attentionScale)
    {
        if (state.totalRows_ == 0U) {
            return;
        }
        if (!state.qScaleReady_) {
            auto q = state.template Ub<float>(BufferLayout::Q_SCALE);
            AscendC::Muls(q, q, attentionScale, state.totalRows_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(state.template Ub<uint32_t>(Q_BROADCAST), q.template ReinterpretCast<uint32_t>(),
                          (state.totalRows_ + 7U) / 8U, AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            state.qScaleReady_ = true;
        }
    }

    __aicore__ inline void Softmax(AscendC::GlobalTensor<int32_t> scores, AscendC::GlobalTensor<int8_t> probability,
                                   const Tile& tile, uint32_t round, float attentionScale)
    {
        if (state.totalRows_ == 0U) {
            return;
        }
        if constexpr (singleRowTile) {
            if (state.totalRows_ == 0U) {
                return;
            }
            state.rowOffset_ = 0U;
            SoftmaxBatch(scores, probability, tile, round, attentionScale);
            return;
        }
        for (state.rowOffset_ = 0U; state.rowOffset_ < state.totalRows_; state.rowOffset_ += BATCH_ROWS) {
            state.rows_ =
                state.totalRows_ - state.rowOffset_ < BATCH_ROWS ? state.totalRows_ - state.rowOffset_ : BATCH_ROWS;
            SoftmaxBatch(scores, probability, tile, round, attentionScale);
        }
    }

    __aicore__ inline void SoftmaxBatch(AscendC::GlobalTensor<int32_t> scores,
                                        AscendC::GlobalTensor<int8_t> probability, const Tile& tile, uint32_t round,
                                        float attentionScale)
    {
        state.columns_ = TileColumns(tile);
        const uint32_t scaleSlot = round % FullQuantLayout::DOUBLE_BUFFER;
        const uint32_t slot = compactUb     ? 0U :
                              singleRowTile ? scaleSlot :
                                              (state.rowOffset_ / BATCH_ROWS) % FullQuantLayout::DOUBLE_BUFFER;
        auto input = state.template Ub<int32_t>(BufferLayout::S_INPUT + slot * ROW_ELEMENTS * sizeof(int32_t));
        auto work = compactUb     ? input.template ReinterpretCast<float>() :
                    singleRowTile ? state.template Ub<float>(BufferLayout::SCORES) :
                                    input.template ReinterpretCast<float>();
        auto p = state.template Ub<int8_t>(BufferLayout::P_OUTPUT + slot * ROW_ELEMENTS);
        // Each AIV emits only its valid rows; Cube ignores padded MMAD rows.
        if constexpr (singleRowTile) {
            AscendC::Duplicate(p.template ReinterpretCast<int32_t>(), 0, ROW_ELEMENTS / sizeof(int32_t));
        }
        if (state.rows_ != 0) {
            if constexpr (!singleRowTile) {
                // O input aliases S; protect the first S DMA after an O update.
                if (state.rowOffset_ == 0U) {
                    state.template Sync<AscendC::HardEvent::V_MTE2>();
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
            }
            AscendC::DataCopy(input, scores[(state.rowStart_ + state.rowOffset_) * state.columns_],
                              state.rows_ * state.columns_);
            // Both transfers use MTE2: this fence completes S and the earlier K-scale DMA.
            state.template Sync<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(work, input, AscendC::RoundMode::CAST_NONE, state.rows_ * state.columns_);
            AscendC::PipeBarrier<PIPE_V>();
            SoftmaxRows(work, tile, round, scaleSlot, attentionScale);
            if constexpr (!singleRowTile) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
            }
            Quantize(work, p, state.rows_ * state.columns_);
            if constexpr (!singleRowTile) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
            }
        }
        state.template Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(probability[(state.rowStart_ + state.rowOffset_) * state.columns_], p,
                          state.rows_ * state.columns_);
        // P is reusable only after MTE3 has consumed the local slot.
        if constexpr (singleRowTile) {
            state.template Sync<AscendC::HardEvent::MTE3_V>();
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
    }

    template <bool isPaged, class PlanType>
    __aicore__ inline void PrefetchExactKScale(AscendC::GlobalTensor<float> scales, const Tile& tile, uint32_t heads,
                                               uint32_t head, uint32_t round, uint64_t kvStart, uint64_t totalKvTokens,
                                               uint32_t blockY, const PlanType& plan)
    {
        auto dst = state.template Ub<float>(BufferLayout::K_SCALE + (round % FullQuantLayout::DOUBLE_BUFFER) *
                                                                        FullQuantLayout::KV_TILE * sizeof(float));
        auto offsets = state.template Ub<uint32_t>(BufferLayout::K_SCALE_OFFSETS);
        const bool packFour = blockY == 4U;
        if (packFour && !state.shortScaleOffsetsReady_) {
            // Every four useful floats occupy one 32-byte DMA slot. This
            // mapping is independent of sparse IDs and is reused by all tasks.
            for (uint32_t i = 0; i < FullQuantLayout::PAGE_TOKENS; ++i) {
                offsets.SetValue(i, (i / 4U * 8U + i % 4U) * sizeof(float));
            }
            state.template Sync<AscendC::HardEvent::S_V>();
            state.shortScaleOffsetsReady_ = true;
        }
        AscendC::Duplicate(dst, 0.0f, FullQuantLayout::KV_TILE);
        state.template Sync<AscendC::HardEvent::V_MTE2>();
        bool needsGather = false;
        for (uint32_t chunk = 0; chunk < FullQuantLayout::PAGES_PER_TILE; ++chunk) {
            const auto& part = tile.pages[chunk];
            if (part.validTokens == 0U) {
                continue;
            }
            if (part.contiguous) {
                const uint64_t source = ScaleOffset<isPaged>(part.sourceToken, heads, head, kvStart, totalKvTokens);
                CopyScaleSpan(dst[chunk * FullQuantLayout::PAGE_TOKENS], scales[source], part.validTokens);
                continue;
            }
            auto scratch =
                state.template Ub<float>(BufferLayout::K_SCALE_GATHER + chunk * BufferLayout::K_SCALE_CHUNK_BYTES);
            uint32_t packed = 0U;
            for (uint32_t ri = part.planBegin; ri < part.planEnd; ++ri) {
                const auto run = plan.GetSpan(ri);
                const uint64_t source = ScaleOffset<isPaged>(run.token, heads, head, kvStart, totalKvTokens);
                if (!packFour) {
                    // Full Y>=8 spans and all preceding destination offsets
                    // are multiples of 32 bytes; only the last tail needs pad.
                    CopyScaleSpan(dst[chunk * FullQuantLayout::PAGE_TOKENS + packed], scales[source], run.length);
                } else {
                    const uint32_t full = run.length / 4U;
                    if (full != 0U) {
                        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(full), 4U * sizeof(float), 0, 0, 0};
                        AscendC::DataCopyPadExtParams<float> pad{true, 0, 4, 0.0f};
                        AscendC::DataCopyPad(scratch[packed * 2U], scales[source], copy, pad);
                    }
                    const uint32_t tail = run.length % 4U;
                    if (tail != 0U) {
                        // The virtual residual is last in compact token order.
                        CopyScaleSpan(scratch[(packed + full * 4U) * 2U], scales[source + full * 4U], tail);
                    }
                }
                packed += run.length;
            }
            needsGather |= packFour;
        }
        if (needsGather) {
            state.template Sync<AscendC::HardEvent::MTE2_V>();
            for (uint32_t chunk = 0; chunk < FullQuantLayout::PAGES_PER_TILE; ++chunk) {
                const auto& part = tile.pages[chunk];
                if (part.validTokens != 0U && !part.contiguous) {
                    auto scratch = state.template Ub<float>(BufferLayout::K_SCALE_GATHER +
                                                            chunk * BufferLayout::K_SCALE_CHUNK_BYTES);
                    AscendC::Gather(dst[chunk * FullQuantLayout::PAGE_TOKENS], scratch, offsets, uint32_t{0},
                                    part.validTokens);
                }
            }
            state.template Sync<AscendC::HardEvent::V_MTE2>();
        }
    }

    __aicore__ inline void CopyScaleSpan(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src,
                                         uint32_t count)
    {
        AscendC::DataCopyExtParams copy{1, count * uint32_t{sizeof(float)}, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>((8U - count % 8U) % 8U), 0.0f};
        AscendC::DataCopyPad(dst, src, copy, pad);
    }

    template <bool isPaged>
    __aicore__ inline uint64_t ScaleOffset(uint64_t token, uint32_t heads, uint32_t head, uint64_t kvStart,
                                           uint64_t totalKvTokens)
    {
        if constexpr (isPaged) {
            return (token / FullQuantLayout::PAGE_TOKENS * heads + head) * FullQuantLayout::PAGE_TOKENS +
                   token % FullQuantLayout::PAGE_TOKENS;
        }
        return static_cast<uint64_t>(head) * totalKvTokens + kvStart + token;
    }

    // Same 512-column, three-level block reduction as the FP16 epilogue.
    // The last mask covers only valid rows, including GQA tails smaller than eight.
    template <bool IS_MAX>
    __aicore__ inline void ReduceRows(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> scores)
    {
        auto scratch = state.template Ub<float>(BufferLayout::P_INT32);
        if ((state.columns_ % 64U) == 0U) {
            state.FullMask();
            if constexpr (IS_MAX) {
                AscendC::BlockReduceMax<float, false>(scratch, scores, state.rows_ * state.columns_ / 64U, uint64_t{0},
                                                      1, 1, 8);
            } else {
                AscendC::BlockReduceSum<float, false>(scratch, scores, state.rows_ * state.columns_ / 64U, uint64_t{0},
                                                      1, 1, 8);
            }
            AscendC::PipeBarrier<PIPE_V>();
            state.RowMask(state.columns_ / 8U);
            if constexpr (IS_MAX) {
                AscendC::BlockReduceMax<float, false>(scratch[BATCH_ROWS * 64U], scratch, state.rows_, uint64_t{0}, 1,
                                                      1, state.columns_ / 64U);
            } else {
                AscendC::BlockReduceSum<float, false>(scratch[BATCH_ROWS * 64U], scratch, state.rows_, uint64_t{0}, 1,
                                                      1, state.columns_ / 64U);
            }
            AscendC::PipeBarrier<PIPE_V>();
            uint64_t mask = 0;
            const uint32_t count = state.rows_ < 8U ? state.rows_ : 8U;
            for (uint32_t row = 0; row < count; ++row) {
                mask |= ((uint64_t{1} << (state.columns_ / 64U)) - 1U) << (row * 8U);
            }
            AscendC::SetVectorMask<int8_t>(0, mask);
            if constexpr (IS_MAX) {
                AscendC::BlockReduceMax<float, false>(dst, scratch[BATCH_ROWS * 64U], (state.rows_ + 7U) / 8U,
                                                      uint64_t{0}, 1, 1, 8);
            } else {
                AscendC::BlockReduceSum<float, false>(dst, scratch[BATCH_ROWS * 64U], (state.rows_ + 7U) / 8U,
                                                      uint64_t{0}, 1, 1, 8);
            }
        } else {
            // A final 32-column fragment uses a masked two-level per-segment tree.
            for (uint32_t column = 0; column < state.columns_; column += 64U) {
                const uint32_t width = state.columns_ - column < 64U ? state.columns_ - column : 64U;
                state.RowMask(width);
                if constexpr (IS_MAX) {
                    AscendC::BlockReduceMax<float, false>(scratch, scores[column], state.rows_, uint64_t{0}, 1, 1,
                                                          state.columns_ / 8U);
                } else {
                    AscendC::BlockReduceSum<float, false>(scratch, scores[column], state.rows_, uint64_t{0}, 1, 1,
                                                          state.columns_ / 8U);
                }
                AscendC::PipeBarrier<PIPE_V>();
                uint64_t mask = 0;
                for (uint32_t row = 0; row < (state.rows_ < 8U ? state.rows_ : 8U); ++row) {
                    mask |= ((uint64_t{1} << (width / 8U)) - 1U) << (row * 8U);
                }
                AscendC::SetVectorMask<int8_t>(0, mask);
                auto result = column == 0U ? dst : scratch[BATCH_ROWS * 64U];
                if constexpr (IS_MAX) {
                    AscendC::BlockReduceMax<float, false>(result, scratch, (state.rows_ + 7U) / 8U, uint64_t{0}, 1, 1,
                                                          8);
                } else {
                    AscendC::BlockReduceSum<float, false>(result, scratch, (state.rows_ + 7U) / 8U, uint64_t{0}, 1, 1,
                                                          8);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (column != 0U) {
                    state.RowMask(state.rows_);
                    if constexpr (IS_MAX) {
                        AscendC::Max(dst, dst, result, state.rows_);
                    } else {
                        AscendC::Add(dst, dst, result, state.rows_);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        state.FullMask();
    }

    __aicore__ inline void SoftmaxRows(AscendC::LocalTensor<float> scores, const Tile& tile, uint32_t round,
                                       uint32_t slot, float attentionScale)
    {
        auto reduceMax = state.template Ub<float>(BufferLayout::REDUCE_MAX);
        auto reduceSum = state.template Ub<float>(BufferLayout::REDUCE_SUM);
        auto scratch = state.template Ub<float>(BufferLayout::P_INT32);
        auto maximum = state.template Ub<float>(BufferLayout::RUNNING_MAX + state.rowOffset_ * sizeof(float));
        auto sum = state.template Ub<float>(BufferLayout::RUNNING_SUM + state.rowOffset_ * sizeof(float));
        auto alpha = state.template Ub<float>(BufferLayout::ALPHA +
                                              (round % FullQuantLayout::STAGES) * BufferLayout::ALPHA_SLOT_BYTES +
                                              state.rowOffset_ * STAT_STRIDE * sizeof(float));
        auto broadcast = scratch[BATCH_ROWS * 72U];
        auto packedAlpha = scratch[BATCH_ROWS * 80U];
        // Initialize reduction padding; scaled Q is cached across KV tiles.
        AscendC::Duplicate(packedAlpha, 0.0f, BATCH_ROWS * 2U);
        AscendC::Duplicate(reduceMax, 0.0f, BATCH_ROWS);
        AscendC::PipeBarrier<PIPE_V>();
        auto broadcastQ = state.template Ub<float>(Q_BROADCAST + state.rowOffset_ * STAT_STRIDE * sizeof(float));
        state.FullMask();
        for (uint32_t column = 0; column < state.columns_; column += 64) {
            state.RowMask(state.columns_ - column < 64U ? state.columns_ - column : 64U);
            AscendC::Mul<float, false>(
                scores[column], scores[column],
                state.template Ub<float>(BufferLayout::K_SCALE +
                                         slot * FullQuantLayout::KV_TILE * sizeof(float))[column],
                uint64_t{0}, state.rows_,
                AscendC::BinaryRepeatParams(1, 1, 1, state.columns_ / 8U, state.columns_ / 8U, 0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < state.columns_; column += 64) {
            state.RowMask(state.columns_ - column < 64U ? state.columns_ - column : 64U);
            AscendC::Mul<float, false>(
                scores[column], scores[column], broadcastQ, uint64_t{0}, state.rows_,
                AscendC::BinaryRepeatParams(1, 1, 0, state.columns_ / 8U, state.columns_ / 8U, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        // One masked vector fill per affected 64-token segment, repeated over all rows.
        for (uint32_t page = 0; page * FullQuantLayout::PAGE_TOKENS < state.columns_; ++page) {
            for (uint32_t column = 0;
                 column < FullQuantLayout::PAGE_TOKENS && page * FullQuantLayout::PAGE_TOKENS + column < state.columns_;
                 column += 64) {
                const uint32_t valid = tile.pages[page].validTokens;
                if (valid < column + 64) {
                    const uint32_t first = valid > column ? valid - column : 0;
                    const uint32_t remaining = state.columns_ - page * FullQuantLayout::PAGE_TOKENS - column;
                    const uint64_t active = remaining >= 64U ? ~uint64_t{0} : ((uint64_t{1} << remaining) - 1U);
                    uint64_t mask[2] = {(~uint64_t{0} << first) & active, 0};
                    AscendC::Duplicate(scores[page * FullQuantLayout::PAGE_TOKENS + column], -3.402823466e38f, mask,
                                       state.rows_, 1, state.columns_ / 8U);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReduceRows<true>(reduceMax, scores);
        state.RowMask(state.rows_);
        AscendC::Max<float, false>(reduceMax, maximum, reduceMax, uint64_t{0}, 1,
                                   AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (round == 0) {
            AscendC::Duplicate(packedAlpha, 0.0f, BATCH_ROWS);
        } else {
            AscendC::Sub<float, false>(packedAlpha, maximum, reduceMax, uint64_t{0}, 1,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<float, false>(packedAlpha, packedAlpha, uint64_t{0}, 1,
                                       AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();
        state.FullMask();
        // Keep the existing three alpha versions and 32-byte row stride for PV consumers.
        AscendC::Brcb(alpha.template ReinterpretCast<uint32_t>(), packedAlpha.template ReinterpretCast<uint32_t>(),
                      BATCH_ROWS / 8U, AscendC::BrcbRepeatParams(1, 8));
        AscendC::Brcb(broadcast.template ReinterpretCast<uint32_t>(), reduceMax.template ReinterpretCast<uint32_t>(),
                      BATCH_ROWS / 8U, AscendC::BrcbRepeatParams(1, 8));
        AscendC::DataCopy(maximum, reduceMax, BATCH_ROWS);
        AscendC::PipeBarrier<PIPE_V>();
        state.FullMask();
        for (uint32_t column = 0; column < state.columns_; column += 64) {
            state.RowMask(state.columns_ - column < 64U ? state.columns_ - column : 64U);
            AscendC::Sub<float, false>(
                scores[column], scores[column], broadcast, uint64_t{0}, state.rows_,
                AscendC::BinaryRepeatParams(1, 1, 0, state.columns_ / 8U, state.columns_ / 8U, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        state.FullMask();
        AscendC::Exp(scores, scores, state.rows_ * state.columns_);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceRows<false>(reduceSum, scores);
        state.RowMask(state.rows_);
        AscendC::Mul<float, false>(sum, sum, packedAlpha, uint64_t{0}, 1,
                                   AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(sum, sum, reduceSum, uint64_t{0}, 1, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        state.FullMask();
    }

    __aicore__ inline void Quantize(AscendC::LocalTensor<float> weights, AscendC::LocalTensor<int8_t> dst,
                                    uint32_t elements)
    {
        AscendC::Muls(weights, weights, state.pScale_, elements);
        AscendC::PipeBarrier<PIPE_V>();
        // Exp(S-rowMax) is in [0,1]; scales in [0,127] need no clipping.
        if (state.pScale_ < 0.0f || state.pScale_ > 127.0f) {
            AscendC::Mins(weights, weights, 127.0f, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(weights, weights, 0.0f, elements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // Reuse the existing wide-row FP32 rounding path for every row tile.
        // Clipped probabilities are in [0,127], so the rounded values are exact in FP16/INT8.
        auto halfValue = state.template Ub<half>(BufferLayout::P_FP16);
        AscendC::Cast(weights, weights, AscendC::RoundMode::CAST_RINT, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(halfValue, weights, AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dst, halfValue, AscendC::RoundMode::CAST_NONE, elements);
    }
};

} // namespace GbsaKernelArch22::Int8

#endif
