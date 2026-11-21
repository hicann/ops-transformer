/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "catlass/catlass.hpp"
namespace Catlass::Epilogue::SmallModelA2 {
// Two UB input slots: issue the next shard before consuming the current shard.
template <class Element>
struct BlockHcPartialReduce {
    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, GM_ADDR partial, GM_ADDR down, GM_ADDR gate, uint32_t rows,
                                   uint32_t r, uint32_t channels, bool combine, uint32_t gateShardStride = 0)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        Reduce(resource, partial, down, rows * r, 0, rows * r);
        if (combine)
            Reduce(resource, partial, gate, rows * channels, static_cast<int64_t>(10) * rows * r,
                   gateShardStride == 0 ? rows * channels : gateShardStride);
    }
    template <class Resource>
    CATLASS_DEVICE void Reduce(Resource& resource, GM_ADDR partial, GM_ADDR dest, uint32_t count, int64_t origin,
                               uint32_t srcStride)
    {
        using namespace AscendC;
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial) + origin);
        GlobalTensor<Element> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(dest));
        auto ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        constexpr uint32_t chunk = 4096;
        auto in = ub.template ReinterpretCast<float>();
        auto sum = ub[32768].template ReinterpretCast<float>();
        auto cast = ub[49152].template ReinterpretCast<Element>();
        // Assign whole 4096-element chunks (8192 bytes for f16/bf16).
        // Consequently no two AIVs own the same output 32-byte block, including
        // the short gate rows; only the final owner handles the padded tail.
        const uint32_t subNum = GetSubBlockNum() == 0 ? 1u : GetSubBlockNum();
        const uint32_t worker = GetBlockIdx();
        const uint32_t workers = GetBlockNum() * subNum;
        for (uint32_t base = worker * chunk; base < count; base += workers * chunk) {
            uint32_t len = count - base < chunk ? count - base : chunk;
            DataCopyExtParams cp{1, len * 4, 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(in, gm[base], cp, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Duplicate(sum, 0.0f, len);
            PipeBarrier<PIPE_V>();
            for (uint32_t k = 0; k < 10; ++k) {
                uint32_t st = k & 1u;
                auto ev = static_cast<event_t>(st);
                if (k + 1 < 10) {
                    uint32_t next = (k + 1) & 1u;
                    if (k > 0)
                        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(next));
                    DataCopyPad(in[next * chunk], gm[static_cast<int64_t>(k + 1) * srcStride + base], cp, pad);
                    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(next));
                }
                WaitFlag<HardEvent::MTE2_V>(ev);
                Add(sum, sum, in[st * chunk], len);
                PipeBarrier<PIPE_V>();
                if (k < 9)
                    SetFlag<HardEvent::V_MTE2>(ev);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            // Last shard consumed input slot1. Drain that V reader before the next
            // chunk or the gate reduction may enqueue MTE2 into the same UB address.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            Cast(cast, sum, RoundMode::CAST_RINT, len);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            cp.blockLen = len * sizeof(Element);
            DataCopyPad(out[base], cast, cp);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace Catlass::Epilogue::SmallModelA2
