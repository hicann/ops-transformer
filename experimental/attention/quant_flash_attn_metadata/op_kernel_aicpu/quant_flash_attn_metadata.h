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
 * \file quant_flash_attn_metadata.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_METADATA_H
#define QUANT_FLASH_ATTN_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t QFA_META_SIZE = 1024;
using QFA_METADATA_T = uint32_t;

constexpr uint32_t QFA_METADATA_SIZE = 16;
constexpr uint32_t QFD_METADATA_SIZE = 16;

// QFA Metadata Index Definitions
constexpr uint32_t QFA_BN2_START_INDEX = 0;
constexpr uint32_t QFA_M_START_INDEX = 1;
constexpr uint32_t QFA_S2_START_INDEX = 2;
constexpr uint32_t QFA_BN2_END_INDEX = 3;
constexpr uint32_t QFA_M_END_INDEX = 4;
constexpr uint32_t QFA_S2_END_INDEX = 5;
constexpr uint32_t QFA_FIRST_QFD_DATA_WORKSPACE_IDX_INDEX = 6;

// QFD Metadata Index Definitions
constexpr uint32_t QFD_BN2_IDX_INDEX = 0;
constexpr uint32_t QFD_M_IDX_INDEX = 1;
constexpr uint32_t QFD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t QFD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t QFD_M_START_INDEX = 4;
constexpr uint32_t QFD_M_NUM_INDEX = 5;

#ifdef __CCE_AICORE__
/**
 * @brief 获取sectionNum的绝对索引
 * @return 返回sectionNum的绝对索引
 */
__aicore__ inline uint32_t GetAttrSectionNumIndex()
{
    return 0U;
}

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t sectionIdx, uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionNum,
                                           bool isAIV = false)
{
    if (isAIV) {
        return sectionNum * AIC_CORE_NUM * QFA_METADATA_SIZE + QFD_METADATA_SIZE * AIV_CORE_NUM * sectionIdx +
               QFD_METADATA_SIZE * coreIdx + metaIdx + 16U;
    } else {
        return QFA_METADATA_SIZE * AIC_CORE_NUM * sectionIdx + QFA_METADATA_SIZE * coreIdx + metaIdx + 16U;
    }
}
#endif

namespace detail {
struct QFaMetaData {
    uint32_t sectionNum;
    uint32_t* qfaMetadata; // [sectionNum][AIC_CORE_NUM][QFA_METADATA_SIZE];
    uint32_t* qfdMetadata; // [sectionNum][AIV_CORE_NUM][QFD_METADATA_SIZE];
    QFaMetaData(void* metadataPtr, uint32_t sectionNum)
        : sectionNum(sectionNum),
          qfaMetadata(static_cast<uint32_t*>(metadataPtr) + 16U),
          qfdMetadata(static_cast<uint32_t*>(metadataPtr) + 16U + sectionNum * AIC_CORE_NUM * QFA_METADATA_SIZE)
    {
        static_cast<uint32_t*>(metadataPtr)[0] = sectionNum;
    }
    void setQFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < QFA_METADATA_SIZE);
        qfaMetadata[AIC_CORE_NUM * QFA_METADATA_SIZE * sectionIdx + QFA_METADATA_SIZE * aicIdx + metaIdx] = val;
    }
    uint32_t getQFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < QFA_METADATA_SIZE);
        return qfaMetadata[AIC_CORE_NUM * QFA_METADATA_SIZE * sectionIdx + QFA_METADATA_SIZE * aicIdx + metaIdx];
    }
    void setQFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < QFD_METADATA_SIZE);
        qfdMetadata[AIV_CORE_NUM * QFD_METADATA_SIZE * sectionIdx + QFD_METADATA_SIZE * aivIdx + metaIdx] = val;
    }
    uint32_t getFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < QFD_METADATA_SIZE);
        return qfdMetadata[AIV_CORE_NUM * QFD_METADATA_SIZE * sectionIdx + QFD_METADATA_SIZE * aivIdx + metaIdx];
    }
};
} // namespace detail

using FA_METADATA_T = uint32_t;

// AICPU metadata format: 16 fields per core (FA and FD both)
constexpr uint32_t METADATA_STRIDE = 16U;
constexpr uint32_t QUANT_FAG_METADATA_SIZE = 4096U;

// Head Metadata Index Definitions
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
constexpr uint32_t QUANT_FAG_DETER_MAX_NUM_INDEX = 0U;

namespace detail {
struct FaMetaData {
    uint32_t sectionNum;
    uint32_t aicNum;
    uint32_t aivNum;
    FA_METADATA_T* headMedata; // [METADATA_STRIDE];
    FA_METADATA_T* faMetadata; // [sectionNum][aicNum][METADATA_STRIDE];
    FA_METADATA_T* fdMetadata; // [sectionNum][aivNum][METADATA_STRIDE];
    FaMetaData(uint32_t aicNum, uint32_t aivNum, uint32_t sectionNum, void* metadataPtr)
        : sectionNum(sectionNum),
          aicNum(aicNum),
          aivNum(aivNum),
          headMedata(static_cast<FA_METADATA_T*>(metadataPtr)),
          faMetadata(headMedata + METADATA_STRIDE),
          fdMetadata(faMetadata + sectionNum * aicNum * METADATA_STRIDE)
    {
        headMedata[0] = sectionNum;
    }

    void Clear()
    {
        for (size_t i = 0; i < METADATA_STRIDE; ++i) {
            headMedata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aicNum * METADATA_STRIDE; ++i) {
            faMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aivNum * METADATA_STRIDE; ++i) {
            fdMetadata[i] = 0U;
        }
    }

    void SetHeadMedata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < METADATA_STRIDE);
        headMedata[metaIdx] = val;
    }

    uint32_t GetHeadMedata(uint32_t metaIdx)
    {
        assert(metaIdx < METADATA_STRIDE);
        return headMedata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicNum);
        assert(metaIdx < METADATA_STRIDE);
        faMetadata[sectionIdx * aicNum * METADATA_STRIDE + aicIdx * METADATA_STRIDE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicNum);
        assert(metaIdx < METADATA_STRIDE);
        return faMetadata[aicNum * METADATA_STRIDE * sectionIdx + METADATA_STRIDE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivNum);
        assert(metaIdx < METADATA_STRIDE);
        fdMetadata[aivNum * METADATA_STRIDE * sectionIdx + METADATA_STRIDE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivNum);
        assert(metaIdx < METADATA_STRIDE);
        return fdMetadata[aivNum * METADATA_STRIDE * sectionIdx + METADATA_STRIDE * aivIdx + metaIdx];
    }
};

struct QuantFAGMetaData {
    uint32_t* data;
    // metadata shape 为 (2, dim0)，第一维存正向 FA 调度数据，
    // 第二维存反向 QuantFAG 调度数据，偏移 dim0 个元素到达第二维起始。
    // dim0 由 torch 层按 sectionNum 最坏值动态分配，从输出 tensor shape 读取。
    QuantFAGMetaData(void* metadataPtr, uint32_t fagOffset)
        : data(static_cast<uint32_t*>(metadataPtr) + fagOffset)
    {}

    void SetDeterMaxRound(uint32_t metaIdx, int64_t val)
    {
        assert(metaIdx < QUANT_FAG_METADATA_SIZE);
        data[metaIdx] = static_cast<uint32_t>(val);
    }

    int64_t GetDeterMaxRound(uint32_t metaIdx)
    {
        assert(metaIdx < QUANT_FAG_METADATA_SIZE);
        return data[metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
