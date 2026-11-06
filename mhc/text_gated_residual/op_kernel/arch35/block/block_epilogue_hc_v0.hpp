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

struct EpilogueHcV0Rms {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = 2;
};

template <class Element, class ArchTag = Arch::Ascend950>
class BlockEpilogueHcV0 {
public:
    using ElementCompute = float;
    static constexpr uint32_t TMP_ELEMS = 512;

    CATLASS_DEVICE
    BlockEpilogueHcV0() {}

    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, GM_ADDR gmX, GM_ADDR gmXn, GM_ADDR gmInject, GM_ADDR gmWGate,
                                   GM_ADDR gmGamma, uint32_t actualM, uint32_t d, uint32_t channels, uint32_t hidden,
                                   float eps, bool useCombine, bool hasGamma, bool disableSubSplit = false,
                                   GM_ADDR gmInvRms = nullptr, bool preferDoubleBuffer = true, bool xnL2Disable = false)
    {
        const uint64_t gammaAll = hasGamma ? static_cast<uint64_t>(channels) * hidden * sizeof(float) : 0u;
        const uint64_t gammaBytes = gammaAll <= 65536u ? gammaAll : (hasGamma ? hidden * sizeof(float) : 0u);
        const uint64_t singleBytes =
            static_cast<uint64_t>(d) * (2u * sizeof(Element) + sizeof(float)) + 8256u + gammaBytes;
        if (singleBytes <= ArchTag::UB_SIZE && channels == 4u && hidden == 2560u) {
            ProcessRows(resource, gmX, gmXn, gmInject, gmWGate, gmGamma, actualM, d, channels, hidden, eps, useCombine,
                        hasGamma, disableSubSplit, gmInvRms, preferDoubleBuffer, xnL2Disable);
            return;
        }
        using namespace AscendC;
        const uint32_t subNum = GetSubBlockNum() == 0 ? 1 : GetSubBlockNum();
        const uint32_t per = (actualM + subNum - 1u) / subNum;
        const uint32_t begin = disableSubSplit ? 0u : GetSubBlockIdx() * per;
        const uint32_t end = disableSubSplit ? actualM : (begin + per < actualM ? begin + per : actualM);
        for (uint32_t row = begin; row < end; ++row) {
            for (uint32_t c = 0; c < channels; ++c) {
                const int64_t off = (static_cast<int64_t>(row) * d + static_cast<int64_t>(c) * hidden) *
                                    static_cast<int64_t>(sizeof(Element));
                ProcessRows(resource, gmX + off, gmXn + off, nullptr, nullptr,
                            hasGamma ? gmGamma + c * hidden * sizeof(float) : nullptr, 1u, hidden, 1u, hidden, eps,
                            false, hasGamma, true,
                            gmInvRms == nullptr ? nullptr :
                                                  gmInvRms + (static_cast<int64_t>(row) * channels + c) *
                                                                 static_cast<int64_t>(sizeof(float)),
                            false, xnL2Disable);
                // Drain the previous channel output before MTE2 reuses its UB storage.
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
            }
        }
    }

    template <class Resource>
    CATLASS_DEVICE void ProcessRows(Resource& resource, GM_ADDR gmX, GM_ADDR gmXn, GM_ADDR gmInject, GM_ADDR gmWGate,
                                    GM_ADDR gmGamma, uint32_t actualM, uint32_t d, uint32_t channels, uint32_t hidden,
                                    float eps, bool useCombine, bool hasGamma, bool disableSubSplit = false,
                                    GM_ADDR gmInvRms = nullptr, bool preferDoubleBuffer = true,
                                    bool xnL2Disable = false)
    {
        using namespace AscendC;
        if (actualM == 0) {
            return;
        }
        const uint32_t subNum = GetSubBlockNum() == 0 ? 1 : GetSubBlockNum();
        const uint32_t subIdx = GetSubBlockIdx();
        const uint32_t row0 = (actualM + subNum - 1) / subNum;
        uint32_t rowStart = subIdx * row0;
        uint32_t rowEnd = rowStart + row0 < actualM ? rowStart + row0 : actualM;
        if (disableSubSplit) {
            rowStart = 0;
            rowEnd = actualM;
        }
        (void)gmInject;
        (void)gmWGate;
        (void)useCombine;
        GlobalTensor<float> gInvRms;
        if (gmInvRms != nullptr) {
            gInvRms.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmInvRms));
        }

        GlobalTensor<Element> gX;
        gX.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmX));
        // x 全量只被本 epilogue 流式读一次（大 shape 可达数百 MiB），关 L2 分配
        // 避免挤掉 C0/C1 权重等高复用数据（arch22 同款优化）。
        gX.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        GlobalTensor<Element> gXn;
        gXn.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmXn));
        if (xnL2Disable) {
            // xn 槽大分支可选关闭 L2：消除 28 核并发的写分配挤占，让位给 C0/C1 权重。
            gXn.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        GlobalTensor<float> gGamma;
        if (hasGamma) {
            gGamma.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmGamma));
        }

        if (rowEnd <= rowStart) {
            return;
        }
        // Whole-D input/output rows use independent ping-pong slots. FP32 reduction
        // scratch is shared because vector computation is sequential.
        auto align32 = [](uint64_t n) { return (n + 31u) & ~uint64_t{31u}; };
        const uint64_t rowBytesT = align32(static_cast<uint64_t>(d) * sizeof(Element));
        const uint64_t rowBytesF = align32(static_cast<uint64_t>(d) * sizeof(float));
        // gamma 很小（C×H×4B），一次搬进 UB 常驻复用（含 +1），省掉逐行逐通道的
        // GM 重搬与 Adds；放不下时退回逐通道搬运的旧路径（arch22 同款优化）。
        const uint32_t gammaAllBytes =
            hasGamma ? align32(static_cast<uint64_t>(channels) * hidden * sizeof(float)) : 0u;
        const bool gammaHoisted = hasGamma && gammaAllBytes <= 65536u;
        const uint64_t gammaBytes =
            hasGamma ? (gammaHoisted ? gammaAllBytes : align32(static_cast<uint64_t>(hidden) * sizeof(float))) : 0u;
        const uint64_t fixedBytes = 16u * sizeof(float) + 2048u * sizeof(float) + gammaBytes;
        const uint64_t doubleBytes = 4u * rowBytesT + rowBytesF + fixedBytes;
        const uint64_t singleBytes = 2u * rowBytesT + rowBytesF + fixedBytes;
        const uint32_t stages = preferDoubleBuffer && doubleBytes <= ArchTag::UB_SIZE ? 2u : 1u;
        // This kernel has no column tiling path. Guard shapes that cannot fit even
        // one input/output pair rather than forming out-of-range UB tensors.
        if (singleBytes > ArchTag::UB_SIZE) {
            return;
        }
        const uint32_t colBytesT = static_cast<uint32_t>(rowBytesT);
        const uint32_t colBytesF = static_cast<uint32_t>(rowBytesF);
        LocalTensor<uint8_t> ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        uint32_t off = 0;
        LocalTensor<Element> inSlot[2];
        LocalTensor<Element> outSlot[2];
        for (uint32_t i = 0; i < stages; ++i) {
            inSlot[i] = ub[off].template ReinterpretCast<Element>();
            off += colBytesT;
        }
        for (uint32_t i = 0; i < stages; ++i) {
            outSlot[i] = ub[off].template ReinterpretCast<Element>();
            off += colBytesT;
        }
        LocalTensor<float> ubRowF = ub[off].template ReinterpretCast<float>();
        off += colBytesF;
        LocalTensor<float> ubMean = ub[off].template ReinterpretCast<float>();
        off += 16u * sizeof(float);
        LocalTensor<float> ubTmp = ub[off].template ReinterpretCast<float>();
        off += 2048u * sizeof(float);
        LocalTensor<float> ubGamma;
        if (hasGamma) {
            ubGamma = ub[off].template ReinterpretCast<float>();
        }
        if (gammaHoisted) {
            for (uint32_t c = 0; c < channels; ++c) {
                HcCopyGmToUb(ubGamma[c * hidden], gGamma[static_cast<int64_t>(c) * hidden], hidden);
            }
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            Adds(ubGamma, ubGamma, 1.0f, channels * hidden);
            PipeBarrier<PIPE_V>();
        }
        const event_t inputEvents[2] = {EVENT_ID0, EVENT_ID1};
        // Separate from inv_rms's EVENT_ID1 in the MTE3/V direction.
        const event_t outputEvents[2] = {EVENT_ID4, EVENT_ID5};
        bool inputReleased[2] = {false, false};
        for (uint32_t i = 0; i < stages; ++i) {
            SetFlag<HardEvent::MTE3_V>(outputEvents[i]);
        }
        HcCopyGmToUb(inSlot[0], gX[static_cast<int64_t>(rowStart) * d], d);
        SetFlag<HardEvent::MTE2_V>(inputEvents[0]);
        for (uint32_t m = rowStart; m < rowEnd; ++m) {
            const uint32_t stage = (m - rowStart) % stages;
            LocalTensor<Element> ubRowT = inSlot[stage];
            WaitFlag<HardEvent::MTE2_V>(inputEvents[stage]);
            // With two slots, prefetch while the current row computes. Reusing an
            // input slot waits for its last vector read, independently of output.
            if (stages == 2u && m + 1u < rowEnd) {
                const uint32_t next = stage ^ 1u;
                if (inputReleased[next]) {
                    WaitFlag<HardEvent::V_MTE2>(inputEvents[next]);
                    inputReleased[next] = false;
                }
                HcCopyGmToUb(inSlot[next], gX[(static_cast<int64_t>(m) + 1u) * d], d);
                SetFlag<HardEvent::MTE2_V>(inputEvents[next]);
            }
            if constexpr (std::is_same_v<Element, float>) {
                Mul(ubRowF, ubRowT, ubRowT, d);
            } else {
                Cast(ubRowF, ubRowT, RoundMode::CAST_NONE, d);
                PipeBarrier<PIPE_V>();
                Mul(ubRowF, ubRowF, ubRowF, d);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < channels; ++c) {
                ReduceSum(ubMean[c * 2], ubRowF[c * hidden], ubTmp, static_cast<int32_t>(hidden));
            }
            PipeBarrier<PIPE_V>();
            Muls(ubMean, ubMean, 1.0f / static_cast<float>(static_cast<int32_t>(hidden)), 16);
            PipeBarrier<PIPE_V>();
            Adds(ubMean, ubMean, eps, 16);
            PipeBarrier<PIPE_V>();
            Rsqrt(ubMean, ubMean, 16);
            PipeBarrier<PIPE_V>();

            const bool saveInvRms = (gmInvRms != nullptr && channels > 0);
            if (saveInvRms) {
                uint64_t rsvdCnt = 0;
                GatherMask(ubTmp, ubMean, static_cast<uint8_t>(1), false, 16, {1, 1, 8, 8}, rsvdCnt);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                DataCopyExtParams invParams;
                invParams.blockCount = 1;
                invParams.blockLen = channels * static_cast<uint32_t>(sizeof(float));
                invParams.srcStride = 0;
                invParams.dstStride = 0;
                DataCopyPad(gInvRms[static_cast<int64_t>(m) * channels], ubTmp, invParams);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }

            if constexpr (std::is_same_v<Element, float>) {
                DataCopy(ubRowF, ubRowT, d);
            } else {
                Cast(ubRowF, ubRowT, RoundMode::CAST_NONE, d);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(inputEvents[stage]);
            inputReleased[stage] = true;
            if (stages == 1u && m + 1u < rowEnd) {
                WaitFlag<HardEvent::V_MTE2>(inputEvents[stage]);
                inputReleased[stage] = false;
                HcCopyGmToUb(inSlot[stage], gX[(static_cast<int64_t>(m) + 1u) * d], d);
                SetFlag<HardEvent::MTE2_V>(inputEvents[stage]);
            }
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
            for (uint32_t c = 0; c < channels; ++c) {
                float scale = ubMean.GetValue(c * 2);
                SetFlag<HardEvent::S_V>(EVENT_ID1);
                WaitFlag<HardEvent::S_V>(EVENT_ID1);
                Muls(ubRowF[c * hidden], ubRowF[c * hidden], scale, hidden);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
            }
            PipeBarrier<PIPE_V>();
            if (hasGamma) {
                for (uint32_t c = 0; c < channels; ++c) {
                    if (gammaHoisted) {
                        // gamma+1 已常驻 UB，按通道偏移直接复用。
                        PipeBarrier<PIPE_V>();
                        Mul(ubRowF[c * hidden], ubRowF[c * hidden], ubGamma[c * hidden], hidden);
                    } else {
                        PipeBarrier<PIPE_ALL>();
                        HcCopyGmToUb(ubGamma, gGamma[static_cast<int64_t>(c) * hidden], hidden);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
                        Adds(ubGamma, ubGamma, 1.0f, hidden);
                        PipeBarrier<PIPE_V>();
                        Mul(ubRowF[c * hidden], ubRowF[c * hidden], ubGamma, hidden);
                    }
                    PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_V>();
            }
            const event_t outEv = outputEvents[stage];
            WaitFlag<HardEvent::MTE3_V>(outEv);
            if constexpr (std::is_same_v<Element, float>) {
                DataCopy(outSlot[stage], ubRowF, d);
            } else {
                Cast(outSlot[stage], ubRowF, RoundMode::CAST_RINT, d);
            }
            SetFlag<HardEvent::V_MTE3>(outEv);
            WaitFlag<HardEvent::V_MTE3>(outEv);
            HcCopyUbToGm(gXn[static_cast<int64_t>(m) * d], outSlot[stage], d);
            SetFlag<HardEvent::MTE3_V>(outEv);
        }
        // Drain both output writes and all input-release tokens before the
        // resource is reused by the next epilogue/stage.
        for (uint32_t i = 0; i < stages; ++i) {
            WaitFlag<HardEvent::MTE3_V>(outputEvents[i]);
            if (inputReleased[i]) {
                WaitFlag<HardEvent::V_MTE2>(inputEvents[i]);
            }
        }
    }
};

} // namespace Catlass::Epilogue
