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
 * \file mixed_quant_flash_attn_metadata.h
 * \brief
 */

#ifndef MIXED_QUANT_FLASH_ATTN_METADATA_H
#define MIXED_QUANT_FLASH_ATTN_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36U;
constexpr uint32_t AIV_CORE_NUM = 72U;

constexpr uint32_t HEAD_METADATA_SIZE = 16U;
constexpr uint32_t FA_METADATA_SIZE = 16U;
constexpr uint32_t FD_METADATA_SIZE = 16U;

// Head Metadata Index Definitions
constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;
constexpr uint32_t HEAD_AIC_CORE_NUM_INDEX = 4U;
constexpr uint32_t HEAD_AIV_CORE_NUM_INDEX = 5U;
constexpr uint32_t HEAD_IS_S1G_INDEX = 6U;

// FA Metadata Index Definitions
constexpr uint32_t FA_BN_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_S2_START_INDEX = 2U;
constexpr uint32_t FA_BN_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
constexpr uint32_t FA_S2_END_INDEX = 5U;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6U;

// FD Metadata Index Definitions
constexpr uint32_t FD_BN_IDX_INDEX = 0U;
constexpr uint32_t FD_M_IDX_INDEX = 1U;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2U;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3U;
constexpr uint32_t FD_M_START_INDEX = 4U;
constexpr uint32_t FD_M_NUM_INDEX = 5U;

namespace detail {
struct FaMetaData {
    uint32_t sectionNum;
    uint32_t aicCoreNum_;
    uint32_t aivCoreNum_;
    uint32_t* headMetadata; // [HEAD_METADATA_SIZE];
    uint32_t* faMetadata;   // [sectionNum][AIC_CORE_NUM][FA_METADATA_SIZE];
    uint32_t* fdMetadata;   // [sectionNum][AIV_CORE_NUM][FD_METADATA_SIZE];
    FaMetaData(void* metadataPtr, uint32_t sectionNum)
        : FaMetaData(metadataPtr, sectionNum, AIC_CORE_NUM, AIV_CORE_NUM)
    {}

    FaMetaData(void* metadataPtr, uint32_t sectionNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
        : sectionNum(sectionNum),
          aicCoreNum_(aicCoreNum),
          aivCoreNum_(aivCoreNum),
          headMetadata(static_cast<uint32_t*>(metadataPtr)),
          faMetadata(headMetadata + HEAD_METADATA_SIZE),
          fdMetadata(faMetadata + sectionNum * aicCoreNum_ * FA_METADATA_SIZE)
    {
        headMetadata[0] = sectionNum;
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < HEAD_METADATA_SIZE);
        headMetadata[metaIdx] = val;
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx)
    {
        assert(metaIdx < HEAD_METADATA_SIZE);
        return headMetadata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicCoreNum_);
        assert(metaIdx < FA_METADATA_SIZE);
        faMetadata[sectionIdx * aicCoreNum_ * FA_METADATA_SIZE + aicIdx * FA_METADATA_SIZE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicCoreNum_);
        assert(metaIdx < FA_METADATA_SIZE);
        return faMetadata[aicCoreNum_ * FA_METADATA_SIZE * sectionIdx + FA_METADATA_SIZE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivCoreNum_);
        assert(metaIdx < FD_METADATA_SIZE);
        fdMetadata[aivCoreNum_ * FD_METADATA_SIZE * sectionIdx + FD_METADATA_SIZE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivCoreNum_);
        assert(metaIdx < FD_METADATA_SIZE);
        return fdMetadata[aivCoreNum_ * FD_METADATA_SIZE * sectionIdx + FD_METADATA_SIZE * aivIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
