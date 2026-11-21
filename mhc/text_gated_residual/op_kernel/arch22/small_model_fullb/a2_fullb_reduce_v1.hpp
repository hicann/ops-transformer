/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "../block/block_epilogue_hc_v1.hpp"

namespace Catlass::Epilogue::SmallModelA2 {
template <class Element>
struct BlockHcReduceV1 {
    template <class Resource>
    CATLASS_DEVICE void Reduce(Resource& resource, GM_ADDR partial, int64_t origin, uint32_t stride, uint32_t offset,
                               uint32_t count)
    {
        using namespace AscendC;
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial) + origin);
        auto ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        auto in = ub.template ReinterpretCast<float>();
        auto sum = ub[32768].template ReinterpretCast<float>();
        DataCopyExtParams cp{1, count * sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(in, gm[offset], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Duplicate(sum, 0.0f, count);
        PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < 10u; ++k) {
            const uint32_t slot = k & 1u;
            const auto ev = static_cast<event_t>(slot);
            if (k + 1u < 10u) {
                const uint32_t next = slot ^ 1u;
                if (k > 0u)
                    WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(next));
                DataCopyPad(in[next * 4096u], gm[static_cast<int64_t>(k + 1u) * stride + offset], cp, pad);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(next));
            }
            WaitFlag<HardEvent::MTE2_V>(ev);
            Add(sum, sum, in[slot * 4096u], count);
            PipeBarrier<PIPE_V>();
            if (k < 9u)
                SetFlag<HardEvent::V_MTE2>(ev);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        auto native = ub[81920].template ReinterpretCast<Element>();
        // Preserve the original reduction -> native C0 output -> FP32 V1 boundary.
        Cast(native, sum, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast(sum, native, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }

    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, GM_ADDR partial, GM_ADDR down, GM_ADDR inject, uint32_t rows,
                                   uint32_t r, uint32_t channels, bool combine, uint32_t gateStride,
                                   GM_ADDR downCache = nullptr)
    {
        using namespace AscendC;
        auto ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        auto sum = ub[32768].template ReinterpretCast<float>();
        auto work = ub[49152].template ReinterpretCast<float>();
        auto dst = ub[65536].template ReinterpretCast<float>();
        auto native = ub[81920].template ReinterpretCast<Element>();
        GlobalTensor<Element> gd, gi;
        gd.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(down));
        gi.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(inject));
        const float invC = 1.0f / static_cast<float>(static_cast<int32_t>(channels));
        // Four C4 native gate rows form one 32B block, owned by a single AIV.
        constexpr uint32_t batch = 4u;
        const uint32_t workers = GetBlockNum() * GetSubBlockNum();
        PipeBarrier<PIPE_ALL>();
        SetMaskNorm();
        ResetMask();
        for (uint32_t base = GetBlockIdx() * batch; base < rows; base += workers * batch) {
            const uint32_t n = rows - base < batch ? rows - base : batch;
            const uint32_t count = n * r;
            Reduce(resource, partial, 0, rows * r, base * r, count);
            if (downCache != nullptr) {
                GlobalTensor<Element> cache;
                cache.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(downCache));
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams cacheCopy{1, count * sizeof(Element), 0, 0, 0};
                DataCopyPad(cache[base * r], native, cacheCopy);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            Muls(sum, sum, invC, count);
            PipeBarrier<PIPE_V>();
            Muls(work, sum, -1.0f, count);
            PipeBarrier<PIPE_V>();
            Exp(work, work, count);
            PipeBarrier<PIPE_V>();
            Adds(work, work, 1.0f, count);
            PipeBarrier<PIPE_V>();
            Div(dst, sum, work, count);
            PipeBarrier<PIPE_V>();
            Cast(native, dst, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams cp{1, count * sizeof(Element), 0, 0, 0};
            DataCopyPad(gd[base * r], native, cp);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if (combine) {
                const uint32_t gateCount = n * channels;
                Reduce(resource, partial, static_cast<int64_t>(10) * rows * r, gateStride, base * channels, gateCount);
                HcInjectCompute(dst, sum, work, gateCount, invC);
                Cast(native, dst, RoundMode::CAST_RINT, gateCount);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                cp.blockLen = gateCount * sizeof(Element);
                DataCopyPad(gi[base * channels], native, cp);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace Catlass::Epilogue::SmallModelA2
