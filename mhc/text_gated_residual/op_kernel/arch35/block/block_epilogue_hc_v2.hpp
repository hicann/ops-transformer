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
#include "../../hc_vector_fallback.hpp"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "../tiles/hc_forward_tiles.hpp"
#include "../hc_gm_copy.hpp"
namespace Catlass::Epilogue {
struct EpilogueHcV2GateMean {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = 2;
};
template <class Element, class ArchTag = Arch::Ascend950, bool EnableRegbase = true>
class BlockEpilogueHcV2 {
public:
    CATLASS_DEVICE BlockEpilogueHcV2() {}
    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, GM_ADDR gmUp, GM_ADDR gmXn, GM_ADDR gmMixed, uint32_t actualM,
                                   uint32_t channels, uint32_t hidden, bool disableSubSplit = false,
                                   GM_ADDR gmUpPreAct = nullptr, bool xnL2Disable = false)
    {
        using namespace AscendC;
        if (actualM == 0 || hidden == 0 || channels == 0)
            return;
        const uint32_t subNum = GetSubBlockNum() == 0 ? 1 : GetSubBlockNum();
        const uint32_t subIdx = GetSubBlockIdx();
        const uint32_t row0 = (actualM + subNum - 1) / subNum;
        uint32_t rowStart = disableSubSplit ? 0 : subIdx * row0;
        uint32_t rowEnd = disableSubSplit ? actualM : (rowStart + row0 < actualM ? rowStart + row0 : actualM);
        if (rowStart >= rowEnd)
            return;
        const float invC = 1.0f / static_cast<float>(static_cast<int32_t>(channels));
        const uint32_t d = channels * hidden;
        auto a32 = [](uint32_t n) { return (n + 31u) & ~31u; };
        const uint32_t inputBytes = a32(d * sizeof(Element));
        const uint32_t fpBytes = a32(d * sizeof(float));
        const uint32_t mixedBytes = a32(hidden * sizeof(float));
        const uint32_t outputBytes = a32(hidden * sizeof(Element));
        // Reuse the up FP32 conversion as sigmoid scratch (-up) in place.
        // FP32 x aliases its current native input slot until tile() completes.
        const uint32_t computeBytes = fpBytes + (std::is_same_v<Element, float> ? 0u : fpBytes + mixedBytes) + 32u;
        const uint32_t slotBytes = 2u * inputBytes + outputBytes;
        constexpr uint32_t kUbBytes = 248u * 1024u;
        const uint32_t slots = computeBytes + 2u * slotBytes <= kUbBytes ? 2u : 1u;
        // Fall back to one input/output slot if needed; oversized single rows need host rejection.
        if (computeBytes + slotBytes > kUbBytes) {
            GlobalTensor<Element> upG, xnG, mixG;
            upG.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmUp));
            xnG.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmXn));
            mixG.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmMixed));
            HcV2Striped<Element>(resource, upG, xnG, mixG, rowStart, rowEnd, channels, hidden);
            return;
        }
        LocalTensor<uint8_t> ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        uint32_t off = 0;
        auto upIn = ub[off].template ReinterpretCast<Element>();
        off += slots * inputBytes;
        auto xIn = ub[off].template ReinterpretCast<Element>();
        off += slots * inputBytes;
        auto tmp = ub[off].template ReinterpretCast<float>();
        off += fpBytes;
        LocalTensor<float> xF;
        LocalTensor<float> mixedF;
        if constexpr (!std::is_same_v<Element, float>) {
            xF = ub[off].template ReinterpretCast<float>();
            off += fpBytes;
            mixedF = ub[off].template ReinterpretCast<float>();
            off += mixedBytes;
        }
        auto ones = ub[off].template ReinterpretCast<float>();
        off += 32u;
        auto output = ub[off].template ReinterpretCast<Element>();
        GlobalTensor<Element> gUp, gX, gMixed;
        gUp.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmUp));
        gX.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmXn));
        gMixed.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmMixed));
        // up 是流式一次性数据（C1 写、V2 读一次即弃），读侧同样关 L2 分配，
        // 避免把 C0/C1 权重等高复用数据挤出 L2（V0 的 gX 同款优化）。
        gUp.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (xnL2Disable) {
            // xn 槽大分支可选关闭 L2 读分配（写侧 V0 同步关闭）；mixed 为 V2 流式
            // 输出无人回读，写侧同样不占 L2。实测二者收益重叠，保留 xn 侧开关。
            gX.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            gMixed.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        // up cache is the C1 destination already, so no V2 cache copy is required.
        (void)gmUpPreAct;
        constexpr event_t ev[2] = {EVENT_ID0, EVENT_ID1};
        PipeBarrier<PIPE_ALL>();
        Duplicate(ones, 1.0f, 8);
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < slots; ++s) {
            SetFlag<HardEvent::V_MTE2>(ev[s]);
            SetFlag<HardEvent::MTE3_V>(ev[s]);
        }
        WaitFlag<HardEvent::V_MTE2>(ev[0]);
        HcCopyGmToUb(upIn, gUp[static_cast<int64_t>(rowStart) * d], d);
        HcCopyGmToUb(xIn, gX[static_cast<int64_t>(rowStart) * d], d);
        SetFlag<HardEvent::MTE2_V>(ev[0]);
        Tile::TileSigmoidMulAcc<ArchTag, Gemm::GemmType<float, layout::RowMajor>, EnableRegbase> tile;
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            const uint32_t s = (row - rowStart) % slots;
            WaitFlag<HardEvent::MTE2_V>(ev[s]);
            auto upT = upIn[s * inputBytes / sizeof(Element)];
            auto xT = xIn[s * inputBytes / sizeof(Element)];
            if constexpr (std::is_same_v<Element, float>) {
                Muls(tmp, upT, -1.0f, d);
                xF = xT;
            } else {
                Cast(tmp, upT, RoundMode::CAST_NONE, d);
                Cast(xF, xT, RoundMode::CAST_NONE, d);
                PipeBarrier<PIPE_V>();
                Muls(tmp, tmp, -1.0f, d);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (!std::is_same_v<Element, float>)
                SetFlag<HardEvent::V_MTE2>(ev[s]);
            // The next row lives in a different slot. One-slot fallback waits until x is consumed.
            if (slots == 2u && row + 1u < rowEnd) {
                const uint32_t nextSlot = (s + 1u) % slots;
                WaitFlag<HardEvent::V_MTE2>(ev[nextSlot]);
                HcCopyGmToUb(upIn[nextSlot * inputBytes / sizeof(Element)], gUp[(static_cast<int64_t>(row) + 1u) * d],
                             d);
                HcCopyGmToUb(xIn[nextSlot * inputBytes / sizeof(Element)], gX[(static_cast<int64_t>(row) + 1u) * d], d);
                SetFlag<HardEvent::MTE2_V>(ev[nextSlot]);
            }
            WaitFlag<HardEvent::MTE3_V>(ev[s]);
            auto dst = output[s * outputBytes / sizeof(Element)];
            if constexpr (std::is_same_v<Element, float>)
                mixedF = dst;
            tile(mixedF, ones, xF, tmp, channels, hidden, invC);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same_v<Element, float>)
                SetFlag<HardEvent::V_MTE2>(ev[s]);
            else {
                Cast(dst, mixedF, RoundMode::CAST_RINT, hidden);
                PipeBarrier<PIPE_V>();
            }
            if (slots == 1u && row + 1u < rowEnd) {
                WaitFlag<HardEvent::V_MTE2>(ev[0]);
                HcCopyGmToUb(upIn, gUp[(static_cast<int64_t>(row) + 1u) * d], d);
                HcCopyGmToUb(xIn, gX[(static_cast<int64_t>(row) + 1u) * d], d);
                SetFlag<HardEvent::MTE2_V>(ev[0]);
            }
            SetFlag<HardEvent::V_MTE3>(ev[s]);
            WaitFlag<HardEvent::V_MTE3>(ev[s]);
            HcCopyUbToGm(gMixed[static_cast<int64_t>(row) * hidden], dst, hidden);
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
