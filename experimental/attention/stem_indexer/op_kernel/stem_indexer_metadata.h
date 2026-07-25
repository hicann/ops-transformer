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
constexpr uint32_t SLI_META_SIZE = 2048U;
using SLI_METADATA_T = uint32_t;

constexpr uint32_t FA_METADATA_SIZE = 8U;
constexpr uint32_t FD_METADATA_SIZE = 8U;

// FA Metadata Index Definitions
constexpr uint32_t SLI_CORE_ENABLE_INDEX = 0U;
constexpr uint32_t SLI_BN2_START_INDEX = 1U;
constexpr uint32_t SLI_M_START_INDEX = 2U;
constexpr uint32_t SLI_S2_START_INDEX = 3U;
constexpr uint32_t SLI_BN2_END_INDEX = 4U;
constexpr uint32_t SLI_M_END_INDEX = 5U;
constexpr uint32_t SLI_S2_END_INDEX = 6U;
constexpr uint32_t SLI_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7U;

// FD Metadata Index Definitions
constexpr uint32_t SLD_CORE_ENABLE_INDEX = 0U;
constexpr uint32_t SLD_BN2_IDX_INDEX = 1U;
constexpr uint32_t SLD_M_IDX_INDEX = 2U;
constexpr uint32_t SLD_WORKSPACE_IDX_INDEX = 3U;
constexpr uint32_t SLD_WORKSPACE_NUM_INDEX = 4U;
constexpr uint32_t SLD_M_START_INDEX = 5U;
constexpr uint32_t SLD_M_NUM_INDEX = 6U;

#ifdef __CCE_AICORE__

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, bool isAIV = false)
{
    if (isAIV) {
        return AIC_CORE_NUM * FA_METADATA_SIZE + FD_METADATA_SIZE * coreIdx + metaIdx;
    } else {
        return FA_METADATA_SIZE * coreIdx + metaIdx;
    }
}
#endif

namespace detail {
struct SliMetadata {
    SLI_METADATA_T *faMetadata; // [AIC_CORE_NUM][FA_METADATA_SIZE];
    SLI_METADATA_T *fdMetadata; // [AIV_CORE_NUM][FD_METADATA_SIZE];
    SliMetadata(void *metadataPtr)
        : faMetadata(static_cast<SLI_METADATA_T *>(metadataPtr)),
          fdMetadata(faMetadata + AIC_CORE_NUM * FA_METADATA_SIZE)
    {
    }
    void SetFaMetadata(uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_SIZE);
        faMetadata[FA_METADATA_SIZE * aicIdx + metaIdx] = val;
    }
    uint32_t GetFaMetadata(uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_SIZE);
        return faMetadata[FA_METADATA_SIZE * aicIdx + metaIdx];
    }
    void SetFdMetadata(uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_SIZE);
        fdMetadata[FD_METADATA_SIZE * aivIdx + metaIdx] = val;
    }
    uint32_t GetFdMetadata(uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_SIZE);
        return fdMetadata[FD_METADATA_SIZE * aivIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif // STEM_INDEXER_METADATA_H
