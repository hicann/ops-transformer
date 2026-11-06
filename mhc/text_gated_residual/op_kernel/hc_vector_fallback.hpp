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
#include "kernel_operator.h"
namespace Catlass::Epilogue {
template <class Element, class Resource>
CATLASS_DEVICE void HcV2Striped(Resource& resource, AscendC::GlobalTensor<Element> up,
                                AscendC::GlobalTensor<Element> xn, AscendC::GlobalTensor<Element> mixed, uint32_t begin,
                                uint32_t end, uint32_t channels, uint32_t hidden)
{
    using namespace AscendC;
    constexpr uint32_t stripe = 256;
    // Each channel uses an independent aligned slot; C is not a UB dimension.
    auto u = resource.ubBuf.template GetBufferByByte<Element>(0);
    auto x = resource.ubBuf.template GetBufferByByte<Element>(stripe * sizeof(Element));
    auto uf = resource.ubBuf.template GetBufferByByte<float>(2 * stripe * sizeof(Element));
    auto xf = resource.ubBuf.template GetBufferByByte<float>(2 * stripe * sizeof(Element) + stripe * sizeof(float));
    auto sum =
        resource.ubBuf.template GetBufferByByte<float>(2 * stripe * sizeof(Element) + 2 * stripe * sizeof(float));
    auto ones =
        resource.ubBuf.template GetBufferByByte<float>(2 * stripe * sizeof(Element) + 3 * stripe * sizeof(float));
    Duplicate(ones, 1.0f, 8);
    PipeBarrier<PIPE_ALL>();
    const int64_t d = static_cast<int64_t>(channels) * hidden;
    for (uint32_t row = begin; row < end; ++row) {
        for (uint32_t h = 0; h < hidden; h += stripe) {
            uint32_t hn = hidden - h < stripe ? hidden - h : stripe;
            Duplicate(sum, 0.0f, hn);
            PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < channels; ++c) {
                int64_t off = static_cast<int64_t>(row) * d + static_cast<int64_t>(c) * hidden + h;
                DataCopyExtParams cp(1, hn * sizeof(Element), 0, 0, 0);
                DataCopyPadExtParams<Element> pad(false, 0, 0, 0);
                DataCopyPad(u, up[off], cp, pad);
                DataCopyPad(x, xn[off], cp, pad);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                Cast(uf, u, RoundMode::CAST_NONE, hn);
                Cast(xf, x, RoundMode::CAST_NONE, hn);
                PipeBarrier<PIPE_V>();
                Muls(uf, uf, -1.0f, hn);
                PipeBarrier<PIPE_V>();
                Exp(uf, uf, hn);
                PipeBarrier<PIPE_V>();
                Adds(uf, uf, 1.0f, hn);
                PipeBarrier<PIPE_V>();
                const BinaryRepeatParams rep{1, 0, 1, 8, 0, 8};
                SetMaskCount();
                SetVectorMask<float, MaskMode::COUNTER>(hn);
                Div<float, false>(uf, ones, uf, MASK_PLACEHOLDER, 1, rep);
                PipeBarrier<PIPE_V>();
                SetMaskNorm();
                ResetMask();
                Mul(uf, uf, xf, hn);
                PipeBarrier<PIPE_V>();
                Muls(uf, uf, 1.0f / static_cast<float>(static_cast<int32_t>(channels)), hn);
                PipeBarrier<PIPE_V>();
                Add(sum, sum, uf, hn);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            Cast(u, sum, RoundMode::CAST_RINT, hn);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyExtParams out(1, hn * sizeof(Element), 0, 0, 0);
            DataCopyPad(mixed[static_cast<int64_t>(row) * hidden + h], u, out);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            PipeBarrier<PIPE_ALL>();
        }
    }
}
} // namespace Catlass::Epilogue
