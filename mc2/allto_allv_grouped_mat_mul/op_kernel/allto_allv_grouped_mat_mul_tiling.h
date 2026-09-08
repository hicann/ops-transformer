/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__
#define __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"

constexpr uint32_t A2AVGMM_MAX_COUNT_NUM = 1024U;
constexpr uint32_t A2AVGMM_MAX_LOCAL_EXPERT_NUM = 512U;
constexpr uint32_t A2AVGMM_MAX_EXPERT_OVERLAP_LOCAL_EXPERT_NUM = A2AVGMM_MAX_LOCAL_EXPERT_NUM;
constexpr uint32_t A2AVGMM_LEGACY_LOCAL_EXPERT_NUM = 128U;
constexpr uint32_t A2AVGMM_EXPERT_OVERLAP_DISABLED = 0U;
constexpr uint32_t A2AVGMM_EXPERT_OVERLAP_ENABLED = 1U;
constexpr uint32_t A2AVGMM_MIN_EXPERT_OVERLAP_TOKENS = 64U;
constexpr uint32_t A2AVGMM_MIN_EXPERT_OVERLAP_DIMENSION = 64U;

namespace AlltoAllvGroupedMatMulAivMode {

#if defined(__CCE_AICORE__)
#define A2AVGMM_AIV_TILING_HOST_DEVICE __aicore__ inline
#else
#define A2AVGMM_AIV_TILING_HOST_DEVICE inline
#endif

struct ExpertOverlapStats {
    uint32_t nonEmptyExperts = 0U;
    uint32_t cappedTotalTokens = 0U;
};

A2AVGMM_AIV_TILING_HOST_DEVICE bool BuildExpertOverlapStats(const int32_t *recvPrefix, uint32_t rankSize,
                                                            uint32_t expertPerRank, ExpertOverlapStats &stats)
{
    stats = {};
    if (recvPrefix == nullptr || rankSize == 0U || expertPerRank == 0U ||
        expertPerRank > A2AVGMM_MAX_LOCAL_EXPERT_NUM ||
        static_cast<uint64_t>(rankSize) * expertPerRank > A2AVGMM_MAX_COUNT_NUM) {
        return false;
    }

    int32_t previousPrefix = 0;
    for (uint32_t expertIdx = 0U; expertIdx < expertPerRank; ++expertIdx) {
        for (uint32_t rankIdx = 0U; rankIdx < rankSize; ++rankIdx) {
            const int32_t currentPrefix = recvPrefix[expertIdx * rankSize + rankIdx];
            if (currentPrefix < previousPrefix) {
                return false;
            }
            previousPrefix = currentPrefix;
        }

        const uint32_t expertEndIndex = expertIdx * rankSize + rankSize - 1U;
        const int32_t expertBegin = expertIdx == 0U ? 0 : recvPrefix[expertEndIndex - rankSize];
        const int32_t expertEnd = recvPrefix[expertEndIndex];
        if (expertEnd < expertBegin) {
            return false;
        }
        const uint32_t cappedExpertTokens =
            static_cast<uint32_t>(expertEnd - expertBegin) > A2AVGMM_MIN_EXPERT_OVERLAP_TOKENS ?
                A2AVGMM_MIN_EXPERT_OVERLAP_TOKENS :
                static_cast<uint32_t>(expertEnd - expertBegin);
        if (cappedExpertTokens > 0U) {
            ++stats.nonEmptyExperts;
        }
        if (stats.cappedTotalTokens < A2AVGMM_MIN_EXPERT_OVERLAP_TOKENS) {
            const uint32_t remaining = A2AVGMM_MIN_EXPERT_OVERLAP_TOKENS - stats.cappedTotalTokens;
            if (cappedExpertTokens >= remaining) {
                stats.cappedTotalTokens = A2AVGMM_MIN_EXPERT_OVERLAP_TOKENS;
            } else {
                stats.cappedTotalTokens += cappedExpertTokens;
            }
        }
    }
    return true;
}

A2AVGMM_AIV_TILING_HOST_DEVICE bool IsExpertOverlapProtocolSafe(const int32_t *recvPrefix, uint32_t rankSize,
                                                                uint32_t expertPerRank)
{
    ExpertOverlapStats stats = {};
    return expertPerRank >= 2U && expertPerRank <= A2AVGMM_MAX_EXPERT_OVERLAP_LOCAL_EXPERT_NUM &&
           BuildExpertOverlapStats(recvPrefix, rankSize, expertPerRank, stats) &&
           (expertPerRank > A2AVGMM_LEGACY_LOCAL_EXPERT_NUM ? stats.nonEmptyExperts != 0U :
                                                              stats.nonEmptyExperts >= 2U);
}

A2AVGMM_AIV_TILING_HOST_DEVICE bool ShouldUseAutomaticExpertOverlap(const int32_t *recvPrefix, uint32_t rankSize,
                                                                    uint32_t expertPerRank, uint32_t k, uint32_t n)
{
    ExpertOverlapStats stats = {};
    return expertPerRank >= 2U && expertPerRank <= A2AVGMM_MAX_EXPERT_OVERLAP_LOCAL_EXPERT_NUM &&
           BuildExpertOverlapStats(recvPrefix, rankSize, expertPerRank, stats) &&
           ((expertPerRank > A2AVGMM_LEGACY_LOCAL_EXPERT_NUM && stats.nonEmptyExperts != 0U) ||
            (stats.nonEmptyExperts >= 2U && stats.cappedTotalTokens >= A2AVGMM_MIN_EXPERT_OVERLAP_TOKENS &&
             k >= A2AVGMM_MIN_EXPERT_OVERLAP_DIMENSION && n >= A2AVGMM_MIN_EXPERT_OVERLAP_DIMENSION));
}

// Nine routed flags, each with at most fifteen outstanding notifications.
// Mixed-core SyncAll uses the runtime-reserved flag 13, outside 0..10.
constexpr uint32_t kExpertReadyFlagCount = 9U;
constexpr uint32_t kExpertReadyCountPerFlag = 15U;
constexpr uint32_t kExpertReadyWindow = kExpertReadyFlagCount * kExpertReadyCountPerFlag;
constexpr uint16_t kSharedReadyFlag = 9U;
constexpr uint16_t kRoutedReadyFlag = 10U;
constexpr uint32_t kCopyFlagBufferBytes = 32U;
constexpr uint32_t kCopyBufferCount = 2U;

A2AVGMM_AIV_TILING_HOST_DEVICE uint16_t ExpertReadyFlagId(uint32_t sequence)
{
    return static_cast<uint16_t>((sequence / kExpertReadyCountPerFlag) % kExpertReadyFlagCount);
}

A2AVGMM_AIV_TILING_HOST_DEVICE bool NeedsExpertEventDrain(uint32_t completedExperts)
{
    return completedExperts != 0U && completedExperts % kExpertReadyWindow == 0U;
}

A2AVGMM_AIV_TILING_HOST_DEVICE uint32_t CopyMoveCapacity(uint64_t ubBytes)
{
    if (ubBytes <= kCopyFlagBufferBytes) {
        return 0U;
    }
    // FP16/BF16: reserve metadata, then divide between two 32-byte-aligned buffers.
    const uint64_t elements = (ubBytes - kCopyFlagBufferBytes) / (kCopyBufferCount * 2U);
    const uint64_t capped = elements < 16384U ? elements : 16384U;
    return static_cast<uint32_t>(capped / 16U * 16U);
}

#undef A2AVGMM_AIV_TILING_HOST_DEVICE

} // namespace AlltoAllvGroupedMatMulAivMode

#pragma pack(push, 8)
struct AlltoAllvGmmInfo {
    uint32_t M = 0U;
    uint32_t K = 0U;
    uint32_t N = 0U;
    uint32_t rankSize = 0U;
    uint32_t expertPerRank = 0U;
    uint32_t maxOutputSize = 0U;
    uint32_t isTransposeB = 0U;
    uint32_t hasSharedExpert = 0U;
    uint32_t hasPermuteOut = 0U;
    uint32_t aivCoreNum = 0U;
    uint32_t aicCoreNum = 0U;
    uint32_t totalUbSize = 0U;
};

struct AlltoAllvGmmCoCTiling {
    uint32_t m0 = 0U;
    uint32_t k0 = 0U;
    uint32_t n0 = 0U;
    uint32_t ubMoveNum = 0U;
    uint32_t swizzlCount = 0U;
    uint32_t swizzlDirect = 0U;
};

struct AlltoAllvGmmExpertMeta {
    uint64_t recvTokenBase = 0U;
    uint32_t tokenCount = 0U;
    uint32_t reserved = 0U;
};

struct AlltoAllvGmmExpertSourceMeta {
    uint64_t dstTokenOffset = 0U;
    uint64_t srcTokenOffset = 0U;
    uint32_t tokenCount = 0U;
    uint32_t sourceRank = 0U;
};

struct AlltoAllvGmmAivTilingData {
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling hcclCcTiling;
    AlltoAllvGmmInfo gmmInfo;
    AlltoAllvGmmInfo mmInfo;
    AlltoAllvGmmCoCTiling gmmCocTiling;
    AlltoAllvGmmCoCTiling mmCocTiling;
    int32_t sendPrefix[A2AVGMM_MAX_COUNT_NUM] = {};
    int32_t recvPrefix[A2AVGMM_MAX_COUNT_NUM] = {};
    uint64_t recvTokenOffset = 0U;
    uint64_t userWorkspaceSize = 0U;
    uint64_t actualWindowBytes = 0U;
    uint64_t requiredWindowBytes = 0U;
    uint64_t payloadBytes = 0U;
    uint64_t countBytes = 0U;
    uint64_t controlBytes = 0U;
    uint32_t countNum = 0U;
    uint32_t is910C = 0U;
    uint32_t expertOverlapMode = A2AVGMM_EXPERT_OVERLAP_DISABLED;
    uint32_t reserved = 0U;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct AlltoAllvGmmTilingData {
    MC2KernelTemplate::HcclA2avTilingInfo hcclA2avTilingInfo;
    MC2KernelTemplate::TaskTilingInfo taskTilingInfo;
    bool isPermuteOut = false;
    bool isNeedMM = false;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData gmmQuantTilingData;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData mmQuantTilingData;
};

// The generated binary advertises one default opParaSize for every tiling key.
// Keep the legacy layout at offset zero, but reserve enough bytes for the AIV
// prefix ABI so runtime tiling is not truncated at the former default size.
static_assert(sizeof(AlltoAllvGmmTilingData) <= sizeof(AlltoAllvGmmAivTilingData),
              "AIV tiling must define the default capacity");
struct AlltoAllvGmmKernelTilingData : public AlltoAllvGmmTilingData {
    uint8_t aivCapacityPadding[sizeof(AlltoAllvGmmAivTilingData) - sizeof(AlltoAllvGmmTilingData)] = {};
};
static_assert(sizeof(AlltoAllvGmmKernelTilingData) == sizeof(AlltoAllvGmmAivTilingData),
              "default tiling capacity must cover the AIV prefix ABI");
#pragma pack(pop)

#endif // __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__
