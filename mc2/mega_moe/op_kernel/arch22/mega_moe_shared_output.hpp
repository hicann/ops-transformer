/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SHARED_OUTPUT_HPP
#define MEGA_MOE_SHARED_OUTPUT_HPP
namespace MegaMoeImpl {
__aicore__ inline void PublishRoutedOutput()
{
    if ASCEND_IS_AIV {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(12);
    }
    if ASCEND_IS_AIC {
        AscendC::CrossCoreWaitFlag<0x2>(12);
    }
}
__aicore__ inline void AddIndependentShared(GM_ADDR y, GM_ADDR shared, uint32_t m, uint32_t h, uint32_t experts)
{
    if ASCEND_IS_AIV {
        // Each FFN result and Yr is BF16. Round the FP32 shared sum to BF16
        // before the final FP32 addition, matching the original branch boundary.
        constexpr uint32_t tile = 1024;
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
        pipe.InitBuffer(storage, tile * (2 * sizeof(bfloat16_t) + 3 * sizeof(float)));
        auto branch = storage.Get<bfloat16_t>();
        auto other = branch[tile];
        auto route = storage.Get<float>()[tile];
        auto sum = route[tile];
        auto temp = sum[tile];
        AscendC::GlobalTensor<bfloat16_t> out, source;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
        source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(shared));
        const uint64_t elements = static_cast<uint64_t>(m) * h;
        const uint32_t core = get_block_idx() + get_subblockid() * get_block_num();
        const uint32_t cores = get_block_num() * get_subblockdim();
        for (uint64_t offset = static_cast<uint64_t>(core) * tile; offset < elements;
             offset += static_cast<uint64_t>(cores) * tile) {
            const uint32_t count = elements - offset < tile ? elements - offset : tile;
            AscendC::DataCopy(branch, out[offset], count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
            AscendC::Cast(route, branch, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Duplicate(sum, 0.0F, count);
            for (uint32_t expert = 0; expert < experts; ++expert) {
                AscendC::DataCopy(other, source[static_cast<uint64_t>(expert) * elements + offset], count);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                AscendC::Cast(temp, other, AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(sum, sum, temp, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
            }
            AscendC::Cast(branch, sum, AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(sum, branch, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(route, route, sum, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(branch, route, AscendC::RoundMode::CAST_RINT, count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::DataCopy(out[offset], branch, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        }
    }
}
} // namespace MegaMoeImpl
#endif
