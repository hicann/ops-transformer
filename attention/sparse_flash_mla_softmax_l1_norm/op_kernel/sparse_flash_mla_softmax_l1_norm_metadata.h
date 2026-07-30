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
 * \file sparse_flash_mla_softmax_l1_norm_metadata.h
 * \brief Runtime metadata consumed by sparse_flash_mla_softmax_l1_norm.
 */
#ifndef SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_H
#define SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_H

#include <cstdint>

namespace optiling {

using SMLA_METADATA_T = int32_t;

constexpr uint32_t SMLA_METADATA_MAX_CORE_NUM = 36;
constexpr uint32_t SMLA_METADATA_SIZE = 64;

namespace detail {
struct SmlaSoftmaxL1NormMetaData {
    int32_t totalNum;
    int32_t formerCoreProcessNum;
    int32_t remainCoreProcessNum;
    int32_t remainCoreNum;
    int32_t totalCoreNum;
};
} // namespace detail

static_assert(SMLA_METADATA_SIZE * sizeof(SMLA_METADATA_T) >= sizeof(detail::SmlaSoftmaxL1NormMetaData));

} // namespace optiling

#endif // SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_H
