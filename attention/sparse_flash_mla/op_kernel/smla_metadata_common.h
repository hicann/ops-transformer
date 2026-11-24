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
 * \file smla_metadata_common.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla / quant_sparse_flash_mla 三个算子共用的
 *        metadata 常量与 分核记录的索引定义。
 */

#ifndef SMLA_METADATA_COMMON_H
#define SMLA_METADATA_COMMON_H

#include <cstdint>

namespace optiling {

// Constants
constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

constexpr int64_t SMLA_METADATA_ALIGNMENT = 1024;

inline constexpr int64_t GetSmlaMetadataSize(uint32_t aicNum, uint32_t aivNum)
{
    const int64_t elements = int64_t{aicNum} * FA_METADATA_SIZE + int64_t{aivNum} * FD_METADATA_SIZE;
    return (elements + SMLA_METADATA_ALIGNMENT - 1) / SMLA_METADATA_ALIGNMENT * SMLA_METADATA_ALIGNMENT;
}

// A MIX block runs one cube with the vector lanes required by its kernel variant.
inline constexpr uint32_t GetSmlaBlockNum(uint32_t cubes, uint32_t vectors, uint32_t vectorsPerBlock)
{
    const uint32_t vectorBlocks = vectors < vectorsPerBlock ? vectors : vectors / vectorsPerBlock;
    return cubes < vectorBlocks ? cubes : vectorBlocks;
}

namespace detail {
using FaMetadataRecord = uint32_t[FA_METADATA_SIZE];
using FdMetadataRecord = uint32_t[FD_METADATA_SIZE];
} // namespace detail

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

#ifdef __CCE_AICORE__
__aicore__ inline uint64_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx)
{
    return uint64_t{coreIdx} * FA_METADATA_SIZE + metaIdx;
}

__aicore__ inline uint64_t GetFdRecordOffset(uint32_t aicNum, uint32_t coreIdx)
{
    return uint64_t{aicNum} * FA_METADATA_SIZE + uint64_t{coreIdx} * FD_METADATA_SIZE;
}
#endif

} // namespace optiling

#endif // SMLA_METADATA_COMMON_H
