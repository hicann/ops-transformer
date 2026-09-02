/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_inplace_index_add_tiling.h
 * \brief
 */
#ifndef MOE_INPLACE_INDEX_ADD_TILING_H
#define MOE_INPLACE_INDEX_ADD_TILING_H
#include <cstdint>

namespace optiling {
struct MoeInplaceIndexAddTilingData {
    int64_t block_num;
    int64_t indices_num;
    int64_t outer_loop;
    int64_t full_num_per_block;
    int64_t tail_num;
    int64_t axis_and_after_data_num_updates;
    int64_t axis_and_after_data_num_var;
    int64_t update_data_num;
    int64_t axis;
    int64_t updates_ub_size;
    int64_t indices_ub_size;
    int64_t tiling_core_num;
    int64_t var_shape_num;
    int64_t updates_shape_num;
};

struct MoeInplaceIndexAddCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    int64_t var_size;
    int64_t var_data_each_block;
    int64_t indices_size;
    int64_t indices_data_each_block;
    int64_t soc_version;
    int64_t atomic_add;
    bool isAscendc{false};
    int64_t coreNum{0};
    int64_t ubSize{0};
};
} // namespace optiling
#endif // MOE_INPLACE_INDEX_ADD_TILING_H
