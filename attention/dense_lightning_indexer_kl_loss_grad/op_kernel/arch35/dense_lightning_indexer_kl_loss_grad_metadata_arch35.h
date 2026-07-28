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
 * \file dense_lightning_indexer_kl_loss_grad_metadata_arch35.h
 * \brief A5 (arch35) metadata definitions for dense_lightning_indexer_kl_loss_grad
 */

#ifndef DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_ARCH35_H
#define DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_ARCH35_H

#include <cstdint>

namespace optiling {
constexpr uint32_t DLIKG_METADATA_SIZE = 64;
using DLIKG_METADATA_T = int32_t;

constexpr uint32_t GRAD_METADATA_SIZE = 6;

constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;

#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx)
{
    return GRAD_METADATA_SIZE * coreIdx + metaIdx;
}
#endif

namespace detail {
struct DlikgMetadata {
    uint32_t gradMetadata[GRAD_METADATA_SIZE];
};
}; // namespace detail

static_assert(DLIKG_METADATA_SIZE * sizeof(DLIKG_METADATA_T) >= sizeof(detail::DlikgMetadata));

}; // namespace optiling

#endif
