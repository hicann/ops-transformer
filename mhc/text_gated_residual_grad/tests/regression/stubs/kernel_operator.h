/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator.h
 * \brief Host-only AscendC event stubs for gradient synchronization regression tests.
 */
// Host-only event instrumentation for the credit protocol regression.
#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "../../../op_kernel/text_gated_residual_grad_constants.h"

constexpr unsigned TEST_AIC = 0;
constexpr unsigned TEST_AIV0 = 1;
constexpr unsigned TEST_AIV1 = 2;
constexpr unsigned TEST_CORE_COUNT = 3;
constexpr unsigned TEST_FLAG_COUNT = 32;
constexpr unsigned TEST_AIV_FLAG_OFFSET = 16;
constexpr unsigned TEST_A5_SYNC_FLAG_BEGIN = 12;
constexpr unsigned TEST_A5_SYNC_FLAG_END = 13;
constexpr unsigned TEST_A2_SYNC_FLAG_BEGIN = 11;
constexpr unsigned TEST_AIV_PER_AIC = 2;
constexpr unsigned TEST_SAME_KIND_MODE = 0;

#define ASCEND_IS_AIC (testCore == TEST_AIC)
inline unsigned testCore = TEST_AIC; // AIC, AIV0, AIV1
inline int testCredits[TEST_CORE_COUNT][TEST_FLAG_COUNT]{};
// Opaque device byte address; the host credit model does not dereference it.
using GM_ADDR = std::byte*;
enum {
    PIPE_FIX,
    PIPE_MTE3,
    PIPE_MTE1,
    PIPE_MTE2
};
inline uint32_t GetBlockIdx()
{
    return testCore == TEST_AIV1 ? 1 : 0;
}

namespace AscendC {
template <unsigned Mode, int Pipe>
void CrossCoreSetFlag(uint16_t flag)
{
    if constexpr (Mode == TEST_SAME_KIND_MODE) {
        return; // Same-kind barriers are outside this single-core-pair model.
    }
    assert(flag < TEST_FLAG_COUNT);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    assert(flag != TEST_A5_SYNC_FLAG_BEGIN && flag != TEST_A5_SYNC_FLAG_END &&
           flag != TEST_A5_SYNC_FLAG_BEGIN + TEST_AIV_FLAG_OFFSET &&
           flag != TEST_A5_SYNC_FLAG_END + TEST_AIV_FLAG_OFFSET);
    if (testCore == TEST_AIC) {
        ++testCredits[flag < TEST_AIV_FLAG_OFFSET ? TEST_AIV0 : TEST_AIV1][flag % TEST_AIV_FLAG_OFFSET];
    } else {
        ++testCredits[TEST_AIC][flag + (testCore == TEST_AIV1 ? TEST_AIV_FLAG_OFFSET : 0)];
    }
#else
    assert(flag < TEST_A2_SYNC_FLAG_BEGIN);
    if (testCore == TEST_AIC) {
        ++testCredits[TEST_AIV0][flag];
        ++testCredits[TEST_AIV1][flag];
    } else {
        ++testCredits[TEST_AIC][flag];
    }
#endif
}

template <unsigned Mode, int Pipe = 0>
void CrossCoreWaitFlag(uint16_t flag)
{
    if constexpr (Mode == TEST_SAME_KIND_MODE) {
        return;
    }
    int count = 1;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
    count = testCore == TEST_AIC ? TEST_AIV_PER_AIC : 1;
#endif
    assert(testCredits[testCore][flag] >= count); // No wait before its producer.
    testCredits[testCore][flag] -= count;
}
} // namespace AscendC
