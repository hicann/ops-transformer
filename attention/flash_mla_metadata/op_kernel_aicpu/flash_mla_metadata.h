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
 * \file flash_mla_metadata.h
 * \brief
 */

#ifndef FLASH_MLA_METADATA_H
#define FLASH_MLA_METADATA_H

#include <cstdint>
#include "../../flash_mla/op_kernel/flash_mla_shared.h"
#include <cassert>

namespace optiling {

using FA_METADATA_T = uint32_t;

static_assert(FlashMlaCommon::FA_METADATA_STRIDE == 16U && FlashMlaCommon::FD_METADATA_STRIDE == 16U &&
                  FlashMlaCommon::HEAD_METADATA_STRIDE == 16U,
              "flash_mla metadata layout must stay 16 words/core");

constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;
constexpr uint32_t HEAD_AIC_NUM_INDEX = 4U;
constexpr uint32_t HEAD_AIV_NUM_INDEX = 5U;
constexpr uint32_t HEAD_OUTPUT_LAYOUT_INDEX = 6U;

// Tile sizes in the header are selected by SoC through flash_mla/op_kernel/utils/flash_mla_config.h.

constexpr uint32_t FA_BN2_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_S2_START_INDEX = 2U;
constexpr uint32_t FA_BN2_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
constexpr uint32_t FA_S2_END_INDEX = 5U;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6U;

constexpr uint32_t FD_BN2_IDX_INDEX = 0U;
constexpr uint32_t FD_M_IDX_INDEX = 1U;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2U;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3U;
constexpr uint32_t FD_M_START_INDEX = 4U;
constexpr uint32_t FD_M_NUM_INDEX = 5U;

namespace detail {
struct FaMetadata {
    uint32_t sectionNum;
    uint32_t aicCoreNum;         // 每 section 的 FA 区核数（硬件平台值，host 透传）
    uint32_t aivCoreNum;         // 每 section 的 FD 区核数（硬件平台值，host 透传）
    FA_METADATA_T* headMetadata; // [HEAD_METADATA_STRIDE];
    FA_METADATA_T* faMetadata;   // [sectionNum][aicCoreNum][FA_METADATA_STRIDE];
    FA_METADATA_T* fdMetadata;   // [sectionNum][aivCoreNum][FD_METADATA_STRIDE];
    FaMetadata(void* metadataPtr, uint32_t sectionNum, uint32_t aicNum, uint32_t aivNum)
        : sectionNum(sectionNum),
          aicCoreNum(aicNum),
          aivCoreNum(aivNum),
          headMetadata(static_cast<FA_METADATA_T*>(metadataPtr)),
          faMetadata(headMetadata + FlashMlaCommon::HEAD_METADATA_STRIDE),
          fdMetadata(faMetadata + sectionNum * aicCoreNum * FlashMlaCommon::FA_METADATA_STRIDE)
    {
        headMetadata[0] = sectionNum;
    }

    void Clear()
    {
        for (size_t i = 0; i < FlashMlaCommon::HEAD_METADATA_STRIDE; ++i) {
            headMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aicCoreNum * FlashMlaCommon::FA_METADATA_STRIDE; ++i) {
            faMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aivCoreNum * FlashMlaCommon::FD_METADATA_STRIDE; ++i) {
            fdMetadata[i] = 0U;
        }
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < FlashMlaCommon::HEAD_METADATA_STRIDE);
        headMetadata[metaIdx] = val;
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx)
    {
        assert(metaIdx < FlashMlaCommon::HEAD_METADATA_STRIDE);
        return headMetadata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicCoreNum);
        assert(metaIdx < FlashMlaCommon::FA_METADATA_STRIDE);
        faMetadata[sectionIdx * aicCoreNum * FlashMlaCommon::FA_METADATA_STRIDE +
                   aicIdx * FlashMlaCommon::FA_METADATA_STRIDE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicCoreNum);
        assert(metaIdx < FlashMlaCommon::FA_METADATA_STRIDE);
        return faMetadata[aicCoreNum * FlashMlaCommon::FA_METADATA_STRIDE * sectionIdx +
                          FlashMlaCommon::FA_METADATA_STRIDE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivCoreNum);
        assert(metaIdx < FlashMlaCommon::FD_METADATA_STRIDE);
        fdMetadata[aivCoreNum * FlashMlaCommon::FD_METADATA_STRIDE * sectionIdx +
                   FlashMlaCommon::FD_METADATA_STRIDE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivCoreNum);
        assert(metaIdx < FlashMlaCommon::FD_METADATA_STRIDE);
        return fdMetadata[aivCoreNum * FlashMlaCommon::FD_METADATA_STRIDE * sectionIdx +
                          FlashMlaCommon::FD_METADATA_STRIDE * aivIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
