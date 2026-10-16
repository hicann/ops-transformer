/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_metadata.h
 * \brief QuantFlashMlaMetadata metadata内存布局定义
 */

#ifndef QUANT_FLASH_MLA_METADATA_H
#define QUANT_FLASH_MLA_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

using FA_METADATA_T = uint32_t;

// AICPU metadata format: 16 fields per core (FA and FD both)
constexpr uint32_t METADATA_STRIDE = 16U;
constexpr uint32_t HEAD_METADATA_SIZE = 16U;
constexpr uint32_t FA_METADATA_SIZE = 16U;
constexpr uint32_t FD_METADATA_SIZE = 16U;

constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;
constexpr uint32_t HEAD_AIC_NUM_INDEX = 4U;
constexpr uint32_t HEAD_AIV_NUM_INDEX = 5U;
constexpr uint32_t HEAD_NEED_INIT_OUTPUT_INDEX = 15U;

constexpr uint32_t FA_BN_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_S2_START_INDEX = 2U;
constexpr uint32_t FA_BN_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
constexpr uint32_t FA_S2_END_INDEX = 5U;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6U;

constexpr uint32_t FD_BN_IDX_INDEX = 0U;
constexpr uint32_t FD_M_IDX_INDEX = 1U;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2U;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3U;
constexpr uint32_t FD_M_START_INDEX = 4U;
constexpr uint32_t FD_M_NUM_INDEX = 5U;

namespace detail {
struct FaMetaData {
    uint32_t sectionNum;
    uint32_t aicNum;
    uint32_t aivNum;
    FA_METADATA_T* headMedata; // [HEAD_METADATA_SIZE];
    FA_METADATA_T* faMetadata; // [sectionNum][aicNum][FA_METADATA_SIZE];
    FA_METADATA_T* fdMetadata; // [sectionNum][aivNum][FD_METADATA_SIZE];
    FaMetaData(uint32_t aicNum, uint32_t aivNum, uint32_t sectionNum, void* metadataPtr)
        : sectionNum(sectionNum),
          aicNum(aicNum),
          aivNum(aivNum),
          headMedata(static_cast<FA_METADATA_T*>(metadataPtr)),
          faMetadata(headMedata + HEAD_METADATA_SIZE),
          fdMetadata(faMetadata + sectionNum * aicNum * FA_METADATA_SIZE)
    {
        headMedata[0] = sectionNum;
    }

    void Clear()
    {
        for (size_t i = 0; i < HEAD_METADATA_SIZE; ++i) {
            headMedata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aicNum * FA_METADATA_SIZE; ++i) {
            faMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aivNum * FD_METADATA_SIZE; ++i) {
            fdMetadata[i] = 0U;
        }
    }

    void SetHeadMedata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < HEAD_METADATA_SIZE);
        headMedata[metaIdx] = val;
    }

    uint32_t GetHeadMedata(uint32_t metaIdx)
    {
        assert(metaIdx < HEAD_METADATA_SIZE);
        return headMedata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicNum);
        assert(metaIdx < FA_METADATA_SIZE);
        faMetadata[sectionIdx * aicNum * FA_METADATA_SIZE + aicIdx * FA_METADATA_SIZE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicNum);
        assert(metaIdx < FA_METADATA_SIZE);
        return faMetadata[aicNum * FA_METADATA_SIZE * sectionIdx + FA_METADATA_SIZE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivNum);
        assert(metaIdx < FD_METADATA_SIZE);
        fdMetadata[aivNum * FD_METADATA_SIZE * sectionIdx + FD_METADATA_SIZE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivNum);
        assert(metaIdx < FD_METADATA_SIZE);
        return fdMetadata[aivNum * FD_METADATA_SIZE * sectionIdx + FD_METADATA_SIZE * aivIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
