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
 * \file matmul_reduce_scatter_tiling_a2a3.h
 * \brief
 */
#ifndef MC2_MATMUL_REDUCE_SCATTER_TILING_A2A3_H
#define MC2_MATMUL_REDUCE_SCATTER_TILING_A2A3_H

#include "../matmul_reduce_scatter_tiling_base.h"
namespace optiling {
class MatmulReduceScatterTilingFuncA2A3 : public MatmulReduceScatterTilingFuncBase {
public:
    CutResult GetCutResult(MatmulReduceScatterTilingData& tilingData, mc2tiling::TilingArgs& args) override;
    ge::graphStatus CheckValidRank(const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK,
                                   MatmulReduceScatterTilingData* tilingData, gert::TilingContext* context,
                                   uint32_t rankSize) override;
    std::string GetRsConfig(MatmulReduceScatterTilingData& tilingData) override;
};
} // namespace optiling

#endif
