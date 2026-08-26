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
 * \file stem_indexer_metadata.h
 * \brief
 */

#ifndef STEM_INDEXER_METADATA_H
#define STEM_INDEXER_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36U;
constexpr uint32_t AIV_CORE_NUM = 72U;
using SLI_METADATA_T = uint32_t;

constexpr uint32_t HEAD_METADATA_STRIDE = 16U;
constexpr uint32_t FA_METADATA_STRIDE = 16U;
constexpr uint32_t AIV_RESERVED_METADATA_STRIDE = 16U;

constexpr uint64_t GetMetadataRequiredElems(uint32_t sectionNum)
{
    return static_cast<uint64_t>(HEAD_METADATA_STRIDE) +
           static_cast<uint64_t>(sectionNum) * (static_cast<uint64_t>(AIC_CORE_NUM) * FA_METADATA_STRIDE +
                                                static_cast<uint64_t>(AIV_CORE_NUM) * AIV_RESERVED_METADATA_STRIDE);
}

// Head Metadata Index Definitions
constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;

// Section-based FA Metadata Index (0-based, matching AICPU flash_attn_metadata format)
constexpr uint32_t SLI_SEC_BN_START_INDEX = 0U;
constexpr uint32_t SLI_SEC_M_START_INDEX = 1U;
constexpr uint32_t SLI_SEC_S2_START_INDEX = 2U;
constexpr uint32_t SLI_SEC_BN_END_INDEX = 3U;
constexpr uint32_t SLI_SEC_M_END_INDEX = 4U;
constexpr uint32_t SLI_SEC_S2_END_INDEX = 5U;

namespace detail {
struct SliMetadata {
    uint32_t sectionNum;
    SLI_METADATA_T *metadata;
    SLI_METADATA_T *headMetadata; // [HEAD_METADATA_STRIDE];
    SLI_METADATA_T *faMetadata;   // [sectionNum][AIC_CORE_NUM][FA_METADATA_STRIDE];
    SliMetadata(void *metadataPtr, uint32_t sectionNum)
        : sectionNum(sectionNum),
          metadata(static_cast<SLI_METADATA_T *>(metadataPtr)),
          headMetadata(static_cast<SLI_METADATA_T *>(metadataPtr)),
          faMetadata(headMetadata + HEAD_METADATA_STRIDE)
    {
        headMetadata[0] = sectionNum;
    }

    void Clear()
    {
        for (size_t i = 0; i < GetMetadataRequiredElems(sectionNum); ++i) {
            metadata[i] = 0U;
        }
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < HEAD_METADATA_STRIDE);
        headMetadata[metaIdx] = val;
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx)
    {
        assert(metaIdx < HEAD_METADATA_STRIDE);
        return headMetadata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_STRIDE);
        faMetadata[sectionIdx * AIC_CORE_NUM * FA_METADATA_STRIDE + aicIdx * FA_METADATA_STRIDE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_STRIDE);
        return faMetadata[AIC_CORE_NUM * FA_METADATA_STRIDE * sectionIdx + FA_METADATA_STRIDE * aicIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif // STEM_INDEXER_METADATA_H
