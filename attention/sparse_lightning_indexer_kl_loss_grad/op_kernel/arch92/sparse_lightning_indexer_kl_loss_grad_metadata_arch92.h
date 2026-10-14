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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_arch92.h
 * \brief A6 (arch92) metadata definitions for sparse_lightning_indexer_kl_loss_grad
 */

#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_ARCH92_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_ARCH92_H

#include <cstdint>

namespace optiling {
constexpr uint32_t SLIKG_METADATA_SIZE_ARCH92 = 64;
using SLIKG_METADATA_T_ARCH92 = int32_t;

constexpr uint32_t GRAD_METADATA_SIZE_ARCH92 = 5;

constexpr uint32_t TOTAL_NUM_ARCH92 = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM_ARCH92 = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM_ARCH92 = 2;
constexpr uint32_t REMAIN_CORE_NUM_ARCH92 = 3;
constexpr uint32_t USED_CORE_NUM_ARCH92 = 4;

#ifdef __CCE_AICORE__ || (defined(__NPU_ARCH__))
__aicore__ inline uint32_t GetAttrAbsIndexArch92(uint32_t coreIdx, uint32_t metaIdx)
{
    return GRAD_METADATA_SIZE_ARCH92 * coreIdx + metaIdx;
}
#endif

namespace detail {
struct SlikgMetadataArch92 {
    uint32_t gradMetadata[GRAD_METADATA_SIZE_ARCH92];
};
}; // namespace detail

static_assert(SLIKG_METADATA_SIZE_ARCH92 * sizeof(SLIKG_METADATA_T_ARCH92) >= sizeof(detail::SlikgMetadataArch92));

}; // namespace optiling

#endif
