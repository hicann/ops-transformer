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
 * \file sparse_flash_mla_grad_metadata.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_GRAD_METADATA_H
#define SPARSE_FLASH_MLA_GRAD_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_MAX_NUM = 80;
constexpr uint32_t AIV_CORE_MAX_NUM = 80;
constexpr uint32_t SMLAG_METADATA_TOTAL_SIZE = 1024;
using SMLAG_METADATA_T = int32_t;

constexpr uint32_t GRAD_METADATA_SIZE = 7;

// Grad Metadata Index Definitions
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;
constexpr uint32_t MAX_ORI_KV_SIZE = 5;
constexpr uint32_t MAX_CMP_KV_SIZE = 6;

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__ || (defined(__NPU_ARCH__))
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx)
{
    return GRAD_METADATA_SIZE * coreIdx + metaIdx;
}
#endif

namespace detail {
struct SmlagMetadata {
    uint32_t gradMetadata[GRAD_METADATA_SIZE];
};
} // namespace detail

static_assert(SMLAG_METADATA_TOTAL_SIZE * sizeof(SMLAG_METADATA_T) >= sizeof(detail::SmlagMetadata));
} // namespace optiling

#endif // SPARSE_FLASH_MLA_GRAD_METADATA_H
