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
 * \file lightning_indexer_v2_metadata.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_V2_METADATA_H
#define LIGHTNING_INDEXER_V2_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
// Fast payload stays immediately after the dynamic core records.
inline constexpr uint32_t LI_V2_METADATA_FAST_WORDS = 2U + 36U * 4U;
inline constexpr uint32_t LI_V2_METADATA_ALIGNMENT = 1024U;

using LI_V2_METADATA_T = int32_t;

inline constexpr uint32_t LI_V2_METADATA_SIZE = 8;
inline constexpr uint32_t LD_V2_METADATA_SIZE = 8;

// 扩展区起点由实际核数确定，与调用方分配的缓冲区总长度无关
#ifdef __CCE_AICORE__
__aicore__ inline constexpr uint64_t GetLiV2MetadataExtensionOffset(uint32_t aicNum, uint32_t aivNum)
#else
inline constexpr uint64_t GetLiV2MetadataExtensionOffset(uint32_t aicNum, uint32_t aivNum)
#endif
{
    return static_cast<uint64_t>(aicNum) * LI_V2_METADATA_SIZE + static_cast<uint64_t>(aivNum) * LD_V2_METADATA_SIZE;
}

#ifdef __CCE_AICORE__
__aicore__ inline constexpr uint64_t GetLiV2MetadataSize(uint32_t aicNum, uint32_t aivNum)
#else
inline constexpr uint64_t GetLiV2MetadataSize(uint32_t aicNum, uint32_t aivNum)
#endif
{
    const uint64_t usedSize = GetLiV2MetadataExtensionOffset(aicNum, aivNum);
    return (usedSize + LI_V2_METADATA_ALIGNMENT - 1U) / LI_V2_METADATA_ALIGNMENT * LI_V2_METADATA_ALIGNMENT;
}

// Capacity available to fast payloads without extending the aligned output.
#ifdef __CCE_AICORE__
__aicore__ inline constexpr uint64_t GetLiV2MetadataExtensionCapacity(uint32_t aicNum, uint32_t aivNum)
#else
inline constexpr uint64_t GetLiV2MetadataExtensionCapacity(uint32_t aicNum, uint32_t aivNum)
#endif
{
    return GetLiV2MetadataSize(aicNum, aivNum) - GetLiV2MetadataExtensionOffset(aicNum, aivNum);
}

// LI Metadata Index Definitions
inline constexpr uint32_t LI_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t LI_V2_BN2_START_INDEX = 1;
inline constexpr uint32_t LI_V2_M_START_INDEX = 2;
inline constexpr uint32_t LI_V2_S2_START_INDEX = 3;
inline constexpr uint32_t LI_V2_BN2_END_INDEX = 4;
inline constexpr uint32_t LI_V2_M_END_INDEX = 5;
inline constexpr uint32_t LI_V2_S2_END_INDEX = 6;
inline constexpr uint32_t LI_V2_FIRST_LD_V2_DATA_WORKSPACE_IDX_INDEX = 7;

// Fast metadata is produced and consumed by the same package version.
inline constexpr uint32_t LI_V2_FAST_CORE_MAX_NUM = 32U;
inline constexpr uint32_t FAST_ARCH22_CORE_MAX_NUM = 36U;
// Fast indices are relative to the dynamic extension area.
inline constexpr uint32_t LI_V2_FAST_METADATA_OFFSET = 0U;
inline constexpr uint32_t LI_V2_FAST_ENABLE_INDEX = LI_V2_FAST_METADATA_OFFSET;
inline constexpr uint32_t LI_V2_FAST_CORE_NUM_INDEX = LI_V2_FAST_METADATA_OFFSET + 1U;
// The fast header contains two uint32_t words: enable and coreNum.
inline constexpr uint32_t LI_V2_FAST_HEADER_WORDS = 2U;
inline constexpr uint32_t LI_V2_FAST_TILE_START_INDEX = LI_V2_FAST_METADATA_OFFSET + LI_V2_FAST_HEADER_WORDS;
inline constexpr uint32_t LI_V2_FAST_TILE_END_INDEX = LI_V2_FAST_TILE_START_INDEX + LI_V2_FAST_CORE_MAX_NUM;

inline constexpr uint32_t LI_V2_FAST_CORE_INFO_INDEX = LI_V2_FAST_METADATA_OFFSET + 2U;
inline constexpr uint32_t LI_V2_FAST_CORE_INFO_SIZE = 4U;
inline constexpr uint32_t LI_V2_FAST_BN2_START_INDEX = 0U;
inline constexpr uint32_t LI_V2_FAST_M_START_INDEX = 1U;
inline constexpr uint32_t LI_V2_FAST_BN2_END_INDEX = 2U;
inline constexpr uint32_t LI_V2_FAST_M_END_INDEX = 3U;

// LD Metadata Index Definitions
inline constexpr uint32_t LD_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t LD_V2_BN2_IDX_INDEX = 1;
inline constexpr uint32_t LD_V2_M_IDX_INDEX = 2;
inline constexpr uint32_t LD_V2_WORKSPACE_IDX_INDEX = 3;
inline constexpr uint32_t LD_V2_WORKSPACE_NUM_INDEX = 4;
inline constexpr uint32_t LD_V2_M_START_INDEX = 5;
inline constexpr uint32_t LD_V2_M_NUM_INDEX = 6;

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t aicNum, bool isAIV = false)
{
    if (isAIV) {
        return LI_V2_METADATA_SIZE * aicNum + LD_V2_METADATA_SIZE * coreIdx + metaIdx;
    } else {
        return LI_V2_METADATA_SIZE * coreIdx + metaIdx;
    }
}
#endif

namespace detail {
struct LiV2FastMetadata {
    uint32_t enable;
    uint32_t coreNum;
    uint32_t tileStart[LI_V2_FAST_CORE_MAX_NUM];
    uint32_t tileEnd[LI_V2_FAST_CORE_MAX_NUM];
};

struct LiV2FastMetadataArch22 {
    uint32_t enable;
    uint32_t coreNum;
    // Whole Q tiles own all visible K blocks, so K boundaries need no storage.
    uint32_t coreInfo[FAST_ARCH22_CORE_MAX_NUM][LI_V2_FAST_CORE_INFO_SIZE];
};
}; // namespace detail

static_assert(sizeof(detail::LiV2FastMetadata) ==
              (LI_V2_FAST_TILE_END_INDEX + LI_V2_FAST_CORE_MAX_NUM) * sizeof(uint32_t));
static_assert(sizeof(detail::LiV2FastMetadata) <= LI_V2_METADATA_FAST_WORDS * sizeof(uint32_t));
static_assert(sizeof(detail::LiV2FastMetadataArch22) <= LI_V2_METADATA_FAST_WORDS * sizeof(uint32_t));
}; // namespace optiling

#endif // LIGHTNING_INDEXER_V2_METADATA_H
