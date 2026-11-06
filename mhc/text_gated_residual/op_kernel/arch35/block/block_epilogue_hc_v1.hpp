/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#pragma once
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "../tiles/hc_forward_tiles.hpp"
#include "../hc_gm_copy.hpp"
namespace Catlass::Epilogue {
struct EpilogueHcV1DivSilu {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = 2;
};
template <class ArchTag = Arch::Ascend950, bool EnableRegbase = true>
class BlockEpilogueHcV1 {
public:
    static constexpr uint32_t COMPUTE_LENGTH = 8192;
    CATLASS_DEVICE static uint64_t Align32(uint64_t n)
    {
        return (n + 31u) & ~uint64_t{31u};
    }
    CATLASS_DEVICE static uint64_t BatchBytes(uint32_t n, uint32_t slots, uint32_t nWsDown, uint32_t nWsGate,
                                              uint32_t rDim, uint64_t t, bool inject, bool cache, uint64_t injRowBytes)
    {
        const uint64_t di = Align32(static_cast<uint64_t>(n) * nWsDown * t);
        const uint64_t gi = inject ? Align32(static_cast<uint64_t>(n) * nWsGate * t) : 0u;
        const uint64_t df = Align32(static_cast<uint64_t>(n) * rDim * sizeof(float));
        const uint64_t gf = inject ? Align32(static_cast<uint64_t>(n) * nWsGate * sizeof(float)) : 0u;
        const uint64_t out = Align32(static_cast<uint64_t>(n) * rDim * t);
        return slots * (di + gi + out * (cache ? 2u : 1u) + n * injRowBytes) + df + gf + (df > gf ? df : gf);
    }

    template <class Resource, class Element>
    CATLASS_DEVICE void operator()(Resource& resource, GM_ADDR gmC0Down, GM_ADDR gmC0Gate, GM_ADDR gmDown,
                                   GM_ADDR gmInject, uint32_t actualM, uint32_t nWsDown, uint32_t nWsGate,
                                   uint32_t rDim, uint32_t channels, bool useCombine, bool disableSubSplit = false,
                                   GM_ADDR gmDownPreAct = nullptr)
    {
        using namespace AscendC;
        if (actualM == 0 || rDim == 0 || channels == 0)
            return;
        const uint32_t subNum = GetSubBlockNum() == 0 ? 1 : GetSubBlockNum();
        const uint32_t row0 = (actualM + subNum - 1) / subNum;
        const uint32_t rowStart = disableSubSplit ? 0 : GetSubBlockIdx() * row0;
        const uint32_t rowEnd = disableSubSplit ? actualM : (rowStart + row0 < actualM ? rowStart + row0 : actualM);
        if (rowStart >= rowEnd)
            return;
        const uint32_t rows = rowEnd - rowStart;
        const bool inject = useCombine && nWsGate != 0 && gmInject != nullptr;
        const bool cache = gmDownPreAct != nullptr;
        const bool compactGate = inject && nWsGate == channels;
        const float invC = 1.0f / static_cast<float>(channels);
        const uint64_t t = sizeof(Element);
        const uint64_t injRowBytes = inject ? Align32(static_cast<uint64_t>(channels) * t) : 0u;
        // C0 down/gate are separate padded matrices. Compute down is packed [rows,R],
        // gate remains [rows,nWsGate]. Native input/output each have independent slots.
        uint32_t slots =
            BatchBytes(1u, 2u, nWsDown, nWsGate, rDim, t, inject, cache, injRowBytes) <= ArchTag::UB_SIZE ? 2u : 1u;
        if (BatchBytes(1u, slots, nWsDown, nWsGate, rDim, t, inject, cache, injRowBytes) > ArchTag::UB_SIZE)
            return;
        uint32_t rowsOnce = 1u;
        // Odd-width rows cannot be packed into aligned vector destinations safely.
        if ((static_cast<uint64_t>(rDim) * sizeof(Element) & 31u) == 0u) {
            while (rowsOnce < rows && BatchBytes(rowsOnce + 1u, slots, nWsDown, nWsGate, rDim, t, inject, cache,
                                                 injRowBytes) <= ArchTag::UB_SIZE)
                ++rowsOnce;
        }
        const uint32_t diBytes = static_cast<uint32_t>(Align32(static_cast<uint64_t>(rowsOnce) * nWsDown * t));
        const uint32_t giBytes =
            inject ? static_cast<uint32_t>(Align32(static_cast<uint64_t>(rowsOnce) * nWsGate * t)) : 0u;
        const uint32_t dfBytes = static_cast<uint32_t>(Align32(static_cast<uint64_t>(rowsOnce) * rDim * sizeof(float)));
        const uint32_t gfBytes =
            inject ? static_cast<uint32_t>(Align32(static_cast<uint64_t>(rowsOnce) * nWsGate * sizeof(float))) : 0u;
        const uint32_t outBytes = static_cast<uint32_t>(Align32(static_cast<uint64_t>(rowsOnce) * rDim * t));
        LocalTensor<uint8_t> ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        uint32_t off = 0;
        auto inDown = ub[off].template ReinterpretCast<Element>();
        off += slots * diBytes;
        auto inGate = ub[off].template ReinterpretCast<Element>();
        off += slots * giBytes;
        auto src = ub[off].template ReinterpretCast<float>();
        off += dfBytes;
        auto gate = ub[off].template ReinterpretCast<float>();
        off += gfBytes;
        auto work = ub[off].template ReinterpretCast<float>();
        off += dfBytes > gfBytes ? dfBytes : gfBytes;
        auto outDown = ub[off].template ReinterpretCast<Element>();
        off += slots * outBytes;
        auto outCache = ub[off].template ReinterpretCast<Element>();
        off += cache ? slots * outBytes : 0u;
        auto outInject = ub[off].template ReinterpretCast<Element>();
        GlobalTensor<Element> gDownIn, gGateIn, gDown, gCache, gInject;
        gDownIn.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmC0Down));
        gGateIn.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmC0Gate));
        gDown.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmDown));
        gCache.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmDownPreAct));
        // down_pre_act 只服务反向保存（大 shape 15MiB 级），不参与前向复用，关 L2
        // 分配避免挤掉高复用数据（arch22 同款优化）。
        gCache.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gInject.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmInject));
        constexpr event_t ev[2] = {EVENT_ID0, EVENT_ID1};
        PipeBarrier<PIPE_ALL>();
        for (uint32_t s = 0; s < slots; ++s) {
            SetFlag<HardEvent::V_MTE2>(ev[s]);
            SetFlag<HardEvent::MTE3_V>(ev[s]);
        }
        WaitFlag<HardEvent::V_MTE2>(ev[0]);
        const uint32_t firstRows = rows < rowsOnce ? rows : rowsOnce;
        HcCopyGmToUb(inDown, gDownIn[static_cast<int64_t>(rowStart) * nWsDown], firstRows * nWsDown);
        if (inject)
            HcCopyGmToUb(inGate, gGateIn[static_cast<int64_t>(rowStart) * nWsGate], firstRows * nWsGate);
        SetFlag<HardEvent::MTE2_V>(ev[0]);
        uint32_t round = 0;
        Tile::TileDivSilu<ArchTag, Gemm::GemmType<float, layout::RowMajor>, COMPUTE_LENGTH, EnableRegbase> silu;
        for (uint32_t base = rowStart; base < rowEnd; base += rowsOnce, ++round) {
            const uint32_t s = round % slots;
            const uint32_t nRows = rowEnd - base < rowsOnce ? rowEnd - base : rowsOnce;
            const uint32_t count = nRows * rDim;
            const uint32_t gateCount = nRows * nWsGate;
            auto downT = inDown[s * diBytes / sizeof(Element)];
            auto gateT = inGate[s * giBytes / sizeof(Element)];
            WaitFlag<HardEvent::MTE2_V>(ev[s]);
            // Strip C0 padding per row, then operate over the entire contiguous batch.
            for (uint32_t row = 0; row < nRows; ++row) {
                if constexpr (std::is_same_v<Element, float>)
                    Adds(src[row * rDim], downT[row * nWsDown], 0.0f, rDim);
                else
                    Cast(src[row * rDim], downT[row * nWsDown], RoundMode::CAST_NONE, rDim);
            }
            if (inject) {
                if constexpr (std::is_same_v<Element, float>)
                    Adds(gate, gateT, 0.0f, gateCount);
                else
                    Cast(gate, gateT, RoundMode::CAST_NONE, gateCount);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(ev[s]);
            if (base + nRows < rowEnd) {
                const uint32_t next = (round + 1u) % slots;
                const uint32_t nextRows = rowEnd - base - nRows < rowsOnce ? rowEnd - base - nRows : rowsOnce;
                WaitFlag<HardEvent::V_MTE2>(ev[next]);
                HcCopyGmToUb(inDown[next * diBytes / sizeof(Element)],
                             gDownIn[(static_cast<int64_t>(base) + nRows) * nWsDown], nextRows * nWsDown);
                if (inject)
                    HcCopyGmToUb(inGate[next * giBytes / sizeof(Element)],
                                 gGateIn[(static_cast<int64_t>(base) + nRows) * nWsGate], nextRows * nWsGate);
                SetFlag<HardEvent::MTE2_V>(ev[next]);
            }
            WaitFlag<HardEvent::MTE3_V>(ev[s]);
            auto dst = outDown[s * outBytes / sizeof(Element)];
            auto cacheDst = outCache[s * outBytes / sizeof(Element)];
            if (cache) {
                // c0dp 语义：down_pre_act 统一保存 C0 原始输出（未除 C），与大分支 C0 直写一致，
                // 反向自行重算 /C。保存必须在下面 invC 的 Muls 之前做，否则存的是已除 C 的值。
                if constexpr (std::is_same_v<Element, float>)
                    Adds(cacheDst, src, 0.0f, count);
                else
                    Cast(cacheDst, src, RoundMode::CAST_RINT, count);
                PipeBarrier<PIPE_V>();
            }
            Muls(src, src, invC, count);
            PipeBarrier<PIPE_V>();
            if constexpr (!std::is_same_v<Element, float>) {
                Cast(dst, src, RoundMode::CAST_RINT, count);
                PipeBarrier<PIPE_V>();
                Cast(src, dst, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
            }
            silu(src, work, count, invC, false);
            if constexpr (std::is_same_v<Element, float>)
                Adds(dst, src, 0.0f, count);
            else
                Cast(dst, src, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            if (inject) {
                Muls(work, gate, invC, gateCount);
                PipeBarrier<PIPE_V>();
                if constexpr (!std::is_same_v<Element, float>) {
                    auto rounded = gate.template ReinterpretCast<Element>();
                    Cast(rounded, work, RoundMode::CAST_RINT, gateCount);
                    PipeBarrier<PIPE_V>();
                    Cast(work, rounded, RoundMode::CAST_NONE, gateCount);
                    PipeBarrier<PIPE_V>();
                }
                if (EnableRegbase && (gateCount % 64u) == 0u) {
                    Tile::HcGateVf(reinterpret_cast<__ubuf__ float*>(work.GetPhyAddr()), gateCount);
                    PipeBarrier<PIPE_V>();
                } else {
                    Muls(gate, work, -1.0f, gateCount);
                    PipeBarrier<PIPE_V>();
                    Exp(gate, gate, gateCount);
                    PipeBarrier<PIPE_V>();
                    Adds(gate, gate, 1.0f, gateCount);
                    PipeBarrier<PIPE_V>();
                    Duplicate(work, 1.0f, gateCount);
                    PipeBarrier<PIPE_V>();
                    Div(work, work, gate, gateCount);
                    PipeBarrier<PIPE_V>();
                    Muls(work, work, 2.0f, gateCount);
                    PipeBarrier<PIPE_V>();
                }
                // Compact gate rows must be cast as one batch: row starts need not be 32B aligned.
                if (compactGate) {
                    if constexpr (std::is_same_v<Element, float>)
                        Adds(outInject[s * rowsOnce * injRowBytes / sizeof(Element)], work, 0.0f, gateCount);
                    else
                        Cast(outInject[s * rowsOnce * injRowBytes / sizeof(Element)], work, RoundMode::CAST_RINT,
                             gateCount);
                } else
                    for (uint32_t row = 0; row < nRows; ++row) {
                        auto injDst = outInject[(s * rowsOnce + row) * injRowBytes / sizeof(Element)];
                        if constexpr (std::is_same_v<Element, float>)
                            Adds(injDst, work[row * nWsGate], 0.0f, channels);
                        else
                            Cast(injDst, work[row * nWsGate], RoundMode::CAST_RINT, channels);
                    }
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(ev[s]);
            WaitFlag<HardEvent::V_MTE3>(ev[s]);
            HcCopyUbToGm(gDown[static_cast<int64_t>(base) * rDim], dst, count);
            if (cache)
                HcCopyUbToGm(gCache[static_cast<int64_t>(base) * rDim], cacheDst, count);
            if (inject) {
                if (compactGate) {
                    HcCopyUbToGm(gInject[static_cast<int64_t>(base) * channels],
                                 outInject[s * rowsOnce * injRowBytes / sizeof(Element)], nRows * channels);
                } else
                    for (uint32_t row = 0; row < nRows; ++row)
                        HcCopyUbToGm(gInject[(static_cast<int64_t>(base) + row) * channels],
                                     outInject[(s * rowsOnce + row) * injRowBytes / sizeof(Element)], channels);
            }
            SetFlag<HardEvent::MTE3_V>(ev[s]);
        }
        for (uint32_t s = 0; s < slots; ++s) {
            WaitFlag<HardEvent::V_MTE2>(ev[s]);
            WaitFlag<HardEvent::MTE3_V>(ev[s]);
        }
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace Catlass::Epilogue
