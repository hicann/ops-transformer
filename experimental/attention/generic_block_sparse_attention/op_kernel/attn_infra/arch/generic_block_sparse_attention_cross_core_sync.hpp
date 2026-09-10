/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_CROSS_CORE_SYNC_HPP
#define GENERIC_BLOCK_SPARSE_ATTENTION_CROSS_CORE_SYNC_HPP

#include "catlass/arch/cross_core_sync.hpp"

namespace NpuArch::Arch {

using Catlass::Arch::AIC_INTER_BLOCK_BARRIER;
using Catlass::Arch::AIV_INTER_BLOCK_BARRIER;
using Catlass::Arch::AIV_INTER_SUBBLOCK_BARRIER;
using Catlass::Arch::FFTS_MAX_FLAG;
using Catlass::Arch::MAX_REVERSE_DEPTH;
using Catlass::Arch::FlagID;
using CrossCoreFlag = Catlass::Arch::CrossCoreFlag;

template <uint32_t REVERSE_DEPTH = MAX_REVERSE_DEPTH>
using CrossCoreFlagWithReverse = Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH>;

// These IDs are part of the GBSA kernel protocol, not Catlass itself.
constexpr uint8_t CROSS_CORE_SYNC_MODE_4 = 4U;
constexpr FlagID FLAG_ID0 = 0;
constexpr FlagID FLAG_ID1 = 1;
constexpr FlagID FLAG_ID2 = 2;
constexpr FlagID FLAG_ID3 = 3;
constexpr FlagID FLAG_ID4 = 4;
constexpr FlagID FLAG_ID5 = 5;
constexpr FlagID FLAG_ID6 = 6;
constexpr FlagID FLAG_ID16 = 16;
constexpr FlagID FLAG_ID17 = 17;
constexpr FlagID FLAG_ID18 = 18;
constexpr FlagID FLAG_ID19 = 19;
constexpr FlagID FLAG_ID20 = 20;
constexpr FlagID FLAG_ID21 = 21;
constexpr FlagID FLAG_ID22 = 22;

using Catlass::Arch::CrossCoreBarrier;
using Catlass::Arch::CrossCoreSetFlag;
using Catlass::Arch::CrossCoreSetFlagWithReverse;
using Catlass::Arch::CrossCoreWaitFlagWithReverse;

// GBSA historically passed a mode template argument to WaitFlag. Catlass'
// implementation does not need it, so keep the source-compatible adapter.
template <uint8_t MODE = 0, pipe_t PIPE = PIPE_S>
CATLASS_DEVICE inline void CrossCoreWaitFlag(CrossCoreFlag &flag)
{
    (void)MODE;
    (void)sizeof(PIPE);
    Catlass::Arch::CrossCoreWaitFlag(flag);
}

} // namespace NpuArch::Arch

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_CROSS_CORE_SYNC_HPP
