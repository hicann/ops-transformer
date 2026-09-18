/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_inplace_add_utils.h
 * \brief
 */

#ifndef QUANT_GROUPED_MATMUL_INPLACE_ADD_UTILS_H
#define QUANT_GROUPED_MATMUL_INPLACE_ADD_UTILS_H

#include <cstdint>

namespace QuantGroupedMatmulInplaceAdd {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t GROUPLIST_INDEX = 3;
constexpr uint32_t PER_TOKEN_SCALE_INDEX = 5;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t DIM_NUM_2D = 2U;
constexpr uint64_t TILING_KEY = 0;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 0;
constexpr uint32_t ATTR_INDEX_GROUP_SIZE = 1;
} // namespace QuantGroupedMatmulInplaceAdd
#endif
