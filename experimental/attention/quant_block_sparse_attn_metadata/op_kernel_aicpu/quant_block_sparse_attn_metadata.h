/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_block_sparse_attn_metadata.h
 * \brief
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_METADATA_H
#define QUANT_BLOCK_SPARSE_ATTN_METADATA_H

#include <cstdint>
#ifndef __CCE_AICORE__
#include <cassert>
#include <cstddef>
#include <cstring>
#endif

namespace optiling {

constexpr uint32_t AIC_CORE_NUM = 36U;
constexpr uint32_t AIV_CORE_NUM = 72U;
constexpr uint32_t QBSA_AIC_SLOTS_ARCH92 = 72U;
constexpr uint32_t QBSA_AIV_SLOTS_ARCH92 = 72U;
// Metadata reserves 72 AIC slots; arch92 host tiling schedules at most 64 cores.
constexpr uint32_t QBSA_MAX_AIC_CORE_NUM_ARCH92 = 64U;
constexpr uint32_t QBSA_META_SIZE = 2048U;
using QBSA_METADATA_T = uint32_t;

constexpr uint32_t QBSA_HEAD_METADATA_SIZE = 8U;
constexpr uint32_t QBSA_HEAD_SECTION_NUM_INDEX = 0U;
// A6 MX readers use the producer's table stride, independently of launched cores.
constexpr uint32_t QBSA_HEAD_AIC_CORE_NUM_INDEX = 1U;
constexpr uint32_t QBSA_HEAD_AIV_CORE_NUM_INDEX = 2U;
constexpr uint32_t QBSA_HEAD_LAYOUT_TAG_INDEX = 3U;
constexpr uint32_t QBSA_METADATA_LAYOUT_V1 = 0x51425301U;

constexpr uint32_t QBSA_METADATA_SIZE = 8U;
constexpr uint32_t FD_METADATA_SIZE = 8U;

// QBSA Metadata Index Definitions
constexpr uint32_t QBSA_CORE_ENABLE_INDEX = 0U;                 // 当前AIC核是否有负载需要计算
constexpr uint32_t QBSA_BN1_START_INDEX = 1U;                   // 当前AIC核的起始bn1Idx
constexpr uint32_t QBSA_M_START_INDEX = 2U;                     // 当前AIC核的起始s1Idx
constexpr uint32_t QBSA_S2_START_INDEX = 3U;                    // 当前AIC核的起始s2Idx
constexpr uint32_t QBSA_BN1_END_INDEX = 4U;                     // 当前AIC核的终止bn1Idx
constexpr uint32_t QBSA_M_END_INDEX = 5U;                       // 当前AIC核的终止s1Idx
constexpr uint32_t QBSA_S2_END_INDEX = 6U;                      // 当前AIC核的终止s2Idx
constexpr uint32_t QBSA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7U; // 当前AIC核第一份规约任务在FD_DATA_WORKSPACE的idx

#ifndef __CCE_AICORE__
struct MetadataSlotConfig {
    uint32_t aicSlots;
    uint32_t aivSlots;
    uint32_t maxAicCoreNum;
};

// The host resolves isA5 with GetCurNpuArch() == NpuArch::DAV_3510.
// AICPU consumes that decision instead of inferring architecture from a SoC name.
inline MetadataSlotConfig GetMetadataSlotConfig(bool isA5)
{
    if (isA5) {
        return {AIC_CORE_NUM, AIV_CORE_NUM, AIC_CORE_NUM};
    }
    return {QBSA_AIC_SLOTS_ARCH92, QBSA_AIV_SLOTS_ARCH92, QBSA_MAX_AIC_CORE_NUM_ARCH92};
}

// Reserve the maximum layout capacity for callers whose Meta/FakeTensor path
// cannot query an NPU. This is allocation capacity, not the active table stride.
inline uint64_t GetQbsaMetadataOutputCapacity(uint64_t batchSize, uint64_t numHeadsQ)
{
    const auto a5 = GetMetadataSlotConfig(true);
    const auto a6 = GetMetadataSlotConfig(false);
    const uint64_t aicSlots = a5.aicSlots > a6.aicSlots ? a5.aicSlots : a6.aicSlots;
    const uint64_t aivSlots = a5.aivSlots > a6.aivSlots ? a5.aivSlots : a6.aivSlots;
    return QBSA_HEAD_METADATA_SIZE + batchSize * numHeadsQ * aicSlots * QBSA_METADATA_SIZE +
           aivSlots * FD_METADATA_SIZE;
}

namespace detail {
inline bool IsPaddedQueryLayout(const char* layout)
{
    return layout != nullptr && (std::strcmp(layout, "BSND") == 0 || std::strcmp(layout, "BNSD") == 0);
}

inline bool IsSupportedQueryLayout(const char* layout, int64_t quantMode)
{
    if (layout == nullptr) {
        return false;
    }
    const bool tnd = std::strcmp(layout, "TND") == 0;
    return (quantMode == 1 && (tnd || std::strcmp(layout, "NTD") == 0)) ||
           (quantMode == 2 && (tnd || IsPaddedQueryLayout(layout)));
}

struct qbsaMetaData {
    uint32_t sectionNum{0U};
    uint32_t aicCoreSlots{AIC_CORE_NUM};
    uint32_t aivCoreSlots{AIV_CORE_NUM};
    QBSA_METADATA_T* headMetadata; // [QBSA_HEAD_METADATA_SIZE];
    QBSA_METADATA_T* qbsaMetadata; // [sectionNum][AIC_CORE_NUM][QBSA_METADATA_SIZE];
    QBSA_METADATA_T* fdMetadata;   // [AIV_CORE_NUM][FD_METADATA_SIZE];

    explicit qbsaMetaData(void* metadataPtr, uint32_t sectionNumVal,
                          MetadataSlotConfig config = {AIC_CORE_NUM, AIV_CORE_NUM, AIC_CORE_NUM})
        : sectionNum(sectionNumVal),
          aicCoreSlots(config.aicSlots),
          aivCoreSlots(config.aivSlots),
          headMetadata(static_cast<QBSA_METADATA_T*>(metadataPtr)),
          qbsaMetadata(headMetadata + QBSA_HEAD_METADATA_SIZE),
          fdMetadata(qbsaMetadata + sectionNum * aicCoreSlots * QBSA_METADATA_SIZE)
    {}

    void Clear()
    {
        for (size_t i = 0; i < QBSA_HEAD_METADATA_SIZE; ++i) {
            headMetadata[i] = 0U;
        }
        for (size_t i = 0; i < static_cast<size_t>(sectionNum) * aicCoreSlots * QBSA_METADATA_SIZE; ++i) {
            qbsaMetadata[i] = 0U;
        }
        for (size_t i = 0; i < aivCoreSlots * FD_METADATA_SIZE; ++i) {
            fdMetadata[i] = 0U;
        }
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < QBSA_HEAD_METADATA_SIZE);
        headMetadata[metaIdx] = static_cast<QBSA_METADATA_T>(val);
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx) const
    {
        assert(metaIdx < QBSA_HEAD_METADATA_SIZE);
        return static_cast<uint32_t>(headMetadata[metaIdx]);
    }

    void SetQbsaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicCoreSlots);
        assert(metaIdx < QBSA_METADATA_SIZE);
        qbsaMetadata[sectionIdx * aicCoreSlots * QBSA_METADATA_SIZE + aicIdx * QBSA_METADATA_SIZE + metaIdx] =
            static_cast<QBSA_METADATA_T>(val);
    }

    uint32_t GetQbsaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx) const
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicCoreSlots);
        assert(metaIdx < QBSA_METADATA_SIZE);
        return static_cast<uint32_t>(
            qbsaMetadata[sectionIdx * aicCoreSlots * QBSA_METADATA_SIZE + aicIdx * QBSA_METADATA_SIZE + metaIdx]);
    }
};
} // namespace detail
#endif

} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_METADATA_H
