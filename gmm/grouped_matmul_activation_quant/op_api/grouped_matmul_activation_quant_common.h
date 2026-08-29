/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_GROUPED_MATMUL_ACTIVATION_QUANT_COMMON_H_
#define OP_API_GROUPED_MATMUL_ACTIVATION_QUANT_COMMON_H_

#include <cstddef>
#include <cstdint>

namespace gmaq {
constexpr size_t SINGLE_TENSOR_SIZE = 1UL;
constexpr size_t FIRST_TENSOR_INDEX = 0UL;
constexpr size_t X_DIM_NUM = 2UL;
constexpr size_t SCALE_DIM_NUM = 3UL;
constexpr size_t WEIGHT_SCALE_DIM_NUM = 4UL;
constexpr size_t GROUP_LIST_DIM_NUM = 1UL;
constexpr size_t OUT_DIM_NUM = 2UL;
constexpr size_t OUT_SCALE_DIM_NUM = 3UL;
constexpr size_t WEIGHT_LOGICAL_DIM_NUM = 3UL;
constexpr size_t WEIGHT_NZ_DIM_NUM = 5UL;

constexpr int64_t DIM_0 = 0L;
constexpr int64_t DIM_1 = 1L;
constexpr int64_t DIM_2 = 2L;
constexpr int64_t DIM_3 = 3L;
constexpr int64_t X_M_DIM_INDEX = DIM_0;
constexpr int64_t WEIGHT_SCALE_TRANS_N_DIM_INDEX = DIM_1;
constexpr int64_t WEIGHT_SCALE_N_DIM_INDEX = DIM_2;
constexpr int64_t MX_GROUP_SIZE = 64L;
constexpr int64_t MX_SCALE_PAIR = 2L;
constexpr int64_t NZ_C0_DIM = 16L;
constexpr int64_t NZ_LAST_DIM_FP8 = 32L;
constexpr int64_t NZ_LAST_DIM_FP4 = 64L;
constexpr int64_t MXFP4_PACK_FACTOR = 2L;
} // namespace gmaq

#endif // OP_API_GROUPED_MATMUL_ACTIVATION_QUANT_COMMON_H_
