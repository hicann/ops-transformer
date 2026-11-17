/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP4_KERNEL_UTILS_HPP
#define GBSA_MXFP4_KERNEL_UTILS_HPP

#include "kernel_operator.h"
#include "../generic_block_sparse_attention_fd_utils.h"

namespace GbsaKernelArch35::Mxfp4 {

constexpr uint32_t GBSA_QUANT_TYPE_A4C4 = 4U;
constexpr uint32_t PRE_LAUNCH = 2U;
constexpr uint32_t DESCRIPTOR_PRE_LAUNCH = 2U;
constexpr uint32_t KSCALE_STAGES = 2U;
constexpr uint32_t P_STAGES = PRE_LAUNCH + 1U;
constexpr uint32_t STATS_STAGES = PRE_LAUNCH + 2U;
constexpr uint32_t DESCRIPTOR_STAGES = PRE_LAUNCH + DESCRIPTOR_PRE_LAUNCH + 1U;
// Keep the host workspace contract; only DESCRIPTOR_STAGES slots are active.
constexpr uint32_t DESCRIPTOR_WORKSPACE_SLOTS = 12U;
constexpr uint32_t DESCRIPTOR_TOKENS = 128U;

// Mode 4 maps each AIV's local flag to id / id+16 on the paired AIC.
// S and PV flags are bidirectional mailboxes: AIV seeds free, AIC publishes
// ready after FIX, AIV returns free after V. Descriptor and P mailboxes start
// free on AIV; AIV publishes ready and PV returns free after its L1 loads.
// KScale ready/free mailboxes identify two descriptor pairs in a full KV tile.
enum class Flag : uint32_t {
    S0 = 0U,
    S1 = 1U,
    DESCRIPTOR_BASE = 2U,  // 2..6
    P_BASE = 7U,           // 7..9
    PV = 10U,              // 10..11
    KSCALE_BASE = 12U,     // 12..13
    KSCALE_FREE_BASE = 14U // 14..15
};
static_assert(static_cast<uint32_t>(Flag::KSCALE_FREE_BASE) + KSCALE_STAGES == 16U,
              "Cross-core flags must fit one AIV bank");

template <pipe_t Pipe>
__aicore__ inline void CubeWaitBoth(uint32_t flag)
{
    AscendC::CrossCoreWaitFlag<4, Pipe>(flag);
    AscendC::CrossCoreWaitFlag<4, Pipe>(flag + 16U);
}

template <pipe_t Pipe>
__aicore__ inline void CubePublishBoth(uint32_t flag)
{
    AscendC::CrossCoreSetFlag<4, Pipe>(flag);
    AscendC::CrossCoreSetFlag<4, Pipe>(flag + 16U);
}

template <bool IsCausal>
__aicore__ __attribute__((noinline)) inline bool ValidateMetadata(
    __gm__ const GbsaMetadata::Metadata* meta,
    const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling)
{
    return GsaFd::ValidateMetadata<IsCausal, GBSA_QUANT_TYPE_A4C4>(meta, &tiling, AscendC::GetBlockNum());
}

struct KernelParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR sparseBlockIdx;
    GM_ADDR sparseBlockCount;
    GM_ADDR metaData;
    GM_ADDR cuSeqLengths;
    GM_ADDR cuSeqLengthsKv;
    GM_ADDR sequsedQ;
    GM_ADDR sequsedKv;
    GM_ADDR blockTable;
    GM_ADDR qDequantScale;
    GM_ADDR kDequantScale;
    GM_ADDR vDequantScale;
    GM_ADDR pQuantScale;
    GM_ADDR o;
    GM_ADDR softmaxLse;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

struct TaskInfo {
    uint32_t batch;
    uint32_t kvHead;
    uint32_t qStorageToken;
    uint32_t qTokenInBatch;
    uint32_t explicitBlocks;
    uint32_t residualBlock;
    uint32_t kvTokens;
    uint32_t tileCount;
    uint64_t vScaleOffset;
    uint32_t vScalePairs;
    uint64_t qByteOffset;
    uint64_t kvByteOffset;
    uint64_t outputOffset;
    uint64_t lseOffset;
    uint64_t sparseOffset;
};

} // namespace GbsaKernelArch35::Mxfp4

#endif // GBSA_MXFP4_KERNEL_UTILS_HPP
