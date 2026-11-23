/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SHARED_W8_DEQUANT_HPP
#define MEGA_MOE_SHARED_W8_DEQUANT_HPP

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

namespace Catlass::Epilogue::Block::detail {

// Each uint64 scale stores its FP32 bits in the low word. The caller has
// already converted the accumulators into a separate FP32 vector. Split the
// freed C tile into disjoint packed and unpacked regions for GatherMask.
CATLASS_DEVICE void ApplySharedW8Scale(AscendC::LocalTensor<float> values, AscendC::LocalTensor<int32_t> scratch,
                                       AscendC::GlobalTensor<uint64_t> scales, uint32_t channel, uint32_t length,
                                       uint32_t scratchWords, int32_t eventVMte2, int32_t eventMte2V)
{
    // A gather repeat consumes 32 packed scales. H and I are 128-aligned.
    constexpr uint32_t scalesPerRepeat = 32;
    const uint32_t capacity = scratchWords / (3 * scalesPerRepeat) * scalesPerRepeat;
    auto packed = scratch.template ReinterpretCast<uint64_t>();
    auto unpacked = scratch[2 * capacity];
    for (uint32_t base = 0; base < length; base += capacity) {
        const uint32_t count = length - base < capacity ? length - base : capacity;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVMte2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVMte2);
        AscendC::DataCopy(packed, scales[channel + base], count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMte2V);

        AscendC::GatherMaskParams gather{};
        gather.src0BlockStride = 1;
        gather.repeatTimes = count / scalesPerRepeat;
        gather.src0RepeatStride = 8;
        uint64_t gathered = 0;
        AscendC::GatherMask(unpacked, scratch, 1, false, 0, gather, gathered);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(values[base], values[base], unpacked.template ReinterpretCast<float>(), count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

} // namespace Catlass::Epilogue::Block::detail
#endif
