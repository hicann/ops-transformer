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
namespace Catlass::Epilogue {
// Each AIV owns complete rows of one (M tile, hidden tile); channel state stays in UB.
template <class Element, uint32_t NH, class ArchTag = Arch::Ascend950>
class BlockEpilogueHcV2ChannelTile {
public:
    static constexpr uint32_t STRIPE = 16u;
    static constexpr uint32_t ACC_BYTES = 128u * NH * static_cast<int64_t>(sizeof(float));
    static constexpr uint32_t T_BYTES = STRIPE * NH * static_cast<int64_t>(sizeof(Element));
    static constexpr uint32_t F_BYTES = STRIPE * NH * static_cast<int64_t>(sizeof(float));
    static_assert(ACC_BYTES + 3u * T_BYTES + 3u * F_BYTES <= ArchTag::UB_SIZE, "C1 epilogue UB overflow");
    template <class Resource>
    CATLASS_DEVICE void Consume(Resource& resource, GM_ADDR up, GM_ADDR xn, GM_ADDR mixed, uint32_t mo, uint32_t am,
                                uint32_t h0, uint32_t hn, uint32_t c, uint32_t d, uint32_t hidden,
                                uint32_t channels = 4u)
    {
        using namespace AscendC;
        uint32_t subNum = GetSubBlockNum() == 0u ? 1u : GetSubBlockNum();
        uint32_t per = (am + subNum - 1u) / subNum, begin = GetSubBlockIdx() * per,
                 end = begin + per < am ? begin + per : am;
        if (begin >= end)
            return;
        auto acc = resource.ubBuf.template GetBufferByByte<float>(0);
        auto ut = resource.ubBuf.template GetBufferByByte<Element>(ACC_BYTES);
        auto xt = resource.ubBuf.template GetBufferByByte<Element>(ACC_BYTES + T_BYTES);
        auto ot = resource.ubBuf.template GetBufferByByte<Element>(ACC_BYTES + 2u * T_BYTES);
        auto uf = resource.ubBuf.template GetBufferByByte<float>(ACC_BYTES + 3u * T_BYTES);
        auto xf = resource.ubBuf.template GetBufferByByte<float>(ACC_BYTES + 3u * T_BYTES + F_BYTES);
        auto one = resource.ubBuf.template GetBufferByByte<float>(ACC_BYTES + 3u * T_BYTES + 2u * F_BYTES);
        GlobalTensor<Element> gu, gx, gm;
        gu.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(up));
        gx.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(xn));
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(mixed));
        for (uint32_t row = begin; row < end; row += STRIPE) {
            uint32_t rows = end - row < STRIPE ? end - row : STRIPE, count = rows * hn;
            uint32_t offset = (row - begin) * hn;
            DataCopyExtParams cp(rows, hn * static_cast<int64_t>(sizeof(Element)),
                                 (d - hn) * static_cast<int64_t>(sizeof(Element)), 0u, 0u);
            DataCopyPadExtParams<Element> pad(false, 0u, 0u, 0);
            int64_t pos = static_cast<int64_t>(mo + row) * d + static_cast<int64_t>(c) * hidden + h0;
            // Ready is waited on V. Gate MTE2 explicitly so it cannot read
            // this channel's GM result before the producer's FIX store.
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(ut, gu[pos], cp, pad);
            DataCopyPad(xt, gx[pos], cp, pad);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Cast(uf, ut, RoundMode::CAST_NONE, count);
            Cast(xf, xt, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Muls(uf, uf, -1.0f, count);
            PipeBarrier<PIPE_V>();
            Exp(uf, uf, count);
            PipeBarrier<PIPE_V>();
            Adds(uf, uf, 1.0f, count);
            PipeBarrier<PIPE_V>();
            Duplicate(one, 1.0f, count);
            PipeBarrier<PIPE_V>();
            Div(uf, one, uf, count);
            PipeBarrier<PIPE_V>();
            Mul(uf, uf, xf, count);
            PipeBarrier<PIPE_V>();
            Muls(uf, uf, 1.0f / static_cast<float>(channels), count);
            PipeBarrier<PIPE_V>();
            if (c == 0u) {
                Duplicate(acc[offset], 0.0f, count);
                PipeBarrier<PIPE_V>();
            }
            Add(acc[offset], acc[offset], uf, count);
            PipeBarrier<PIPE_V>();
            if (c + 1u == channels) {
                Cast(ot, acc[offset], RoundMode::CAST_RINT, count);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyExtParams out(rows, hn * static_cast<int64_t>(sizeof(Element)), 0u,
                                      (hidden - hn) * static_cast<int64_t>(sizeof(Element)), 0u);
                DataCopyPad(gm[static_cast<int64_t>(mo + row) * hidden + h0], ot, out);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
};
} // namespace Catlass::Epilogue
