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

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue {

// A2 向量 Rsqrt 的结果只有约 8 bit 尾数，过不了 FP32 混合容差。标量牛顿迭代给出完整 FP32。
CATLASS_DEVICE
float HcRsqrtF32(float x)
{
    if (!(x > 0.0f)) {
        return 0.0f;
    }
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = x;
    bits.u = 0x5f375a86u - (bits.u >> 1);
    float y = bits.f;
    float halfX = 0.5f * x;
    y = y * (1.5f - halfX * y * y);
    y = y * (1.5f - halfX * y * y);
    y = y * (1.5f - halfX * y * y);
    y = y * (1.5f - halfX * y * y);
    return y;
}

// Atlas A2：GM 地址和 UB 地址都要 32 字节对齐。长度不是 32 字节时用 DataCopyPad，
// blockLen 写实际有效字节数，不开启 isPad。后续计算按有效长度走，尾部 dummy 不参与。

template <typename T>
CATLASS_DEVICE void HcCopyGmToUb(AscendC::LocalTensor<T> const& dst, AscendC::GlobalTensor<T> const& src, uint32_t n)
{
    using namespace AscendC;
    if (n == 0) {
        return;
    }
    uint32_t bytes = n * static_cast<uint32_t>(sizeof(T));
    if ((bytes % 32u) == 0u) {
        DataCopy(dst, src, n);
        return;
    }
    DataCopyExtParams p{1, bytes, 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(dst, src, p, pad);
}

template <typename T>
CATLASS_DEVICE void HcCopyUbToGm(AscendC::GlobalTensor<T> const& dst, AscendC::LocalTensor<T> const& src, uint32_t n)
{
    using namespace AscendC;
    if (n == 0) {
        return;
    }
    uint32_t bytes = n * static_cast<uint32_t>(sizeof(T));
    if ((bytes % 32u) == 0u) {
        DataCopy(dst, src, n);
        return;
    }
    DataCopyExtParams p{1, bytes, 0, 0, 0};
    DataCopyPad(dst, src, p);
}

// scratch 至少 32 字节且 32 字节对齐，不与 src 重叠。base 必须是 32 字节对齐的张量起点。
template <typename T>
CATLASS_DEVICE void HcStoreGmElems(AscendC::GlobalTensor<T> const& base, uint32_t elemOff,
                                   AscendC::LocalTensor<T> const& src, uint32_t n,
                                   AscendC::LocalTensor<T> const& scratch)
{
    using namespace AscendC;
    if (n == 0) {
        return;
    }
    constexpr uint32_t kElem = static_cast<uint32_t>(sizeof(T));
    constexpr uint32_t kBlockElems = 32u / kElem;
    uint32_t done = 0;
    while (done < n) {
        uint32_t absElem = elemOff + done;
        uint32_t byteOff = absElem * kElem;
        uint32_t alignedByte = byteOff & ~31u;
        uint32_t prefixElems = (byteOff - alignedByte) / kElem;
        uint32_t room = kBlockElems - prefixElems;
        uint32_t chunk = (n - done) < room ? (n - done) : room;
        uint32_t startElem = alignedByte / kElem;
        if (prefixElems == 0 && chunk == kBlockElems) {
            DataCopy(base[startElem], src[done], chunk);
        } else {
            DataCopy(scratch, base[startElem], kBlockElems);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            SetFlag<HardEvent::V_S>(EVENT_ID3);
            WaitFlag<HardEvent::V_S>(EVENT_ID3);
            for (uint32_t i = 0; i < chunk; ++i) {
                scratch.SetValue(prefixElems + i, src.GetValue(done + i));
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
            DataCopy(base[startElem], scratch, kBlockElems);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
        done += chunk;
    }
}

} // namespace Catlass::Epilogue
