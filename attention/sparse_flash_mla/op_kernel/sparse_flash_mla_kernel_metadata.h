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
 * \file sparse_flash_mla_kernel_metadata.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_KERNEL_METADATA_H
#define SPARSE_FLASH_MLA_KERNEL_METADATA_H

#include <cstdint>
#include "arch35/common/smla_metadata_common.h"

namespace optiling {

constexpr uint32_t SMLA_METADATA_TOTAL_SIZE = 1024;
using SMLA_METADATA_T = int32_t;

namespace detail {
struct SmlaMetadata {
    uint32_t faMetadata[AIC_CORE_MAX_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_MAX_NUM][FD_METADATA_SIZE];
};
}; // namespace detail

static_assert(SMLA_METADATA_TOTAL_SIZE * sizeof(SMLA_METADATA_T) >= sizeof(detail::SmlaMetadata));
}; // namespace optiling

#endif // SPARSE_FLASH_MLA_KERNEL_METADATA_H
