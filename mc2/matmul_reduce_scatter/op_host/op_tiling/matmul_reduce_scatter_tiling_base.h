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
 * \file matmul_reduce_scatter_tiling_base.h
 * \brief
 */
#ifndef MC2_MATMUL_REDUCE_SCATTER_TILING_BASE_H
#define MC2_MATMUL_REDUCE_SCATTER_TILING_BASE_H

#include "vector"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_hcom_topo_info.h"
#include "util/math_util.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "matmul_reduce_scatter_v2/op_host/op_tiling/reduce_scatter_formulaic_tiling.h"
#include "../../op_kernel/matmul_reduce_scatter_tiling_key.h"
#include "../../op_kernel/matmul_reduce_scatter_tiling.h"

namespace optiling {
class MatmulReduceScatterTilingFuncBase {
public:
    ge::graphStatus MatmulReduceScatterTilingFunc(gert::TilingContext* context);
    ge::graphStatus SetTilingData(const gert::TilingContext* context, MatmulReduceScatterTilingData& tilingData);
    ge::graphStatus MC2SetWorkspaceReduceScatter(gert::TilingContext* context,
                                                 MatmulReduceScatterTilingData& tilingData,
                                                 mc2tiling::TilingArgs& args);
    ge::graphStatus MCSpliteMReduceScatter(gert::TilingContext* ctx, MatmulReduceScatterTilingData& tilingData,
                                           mc2tiling::TilingArgs& args);
    ge::graphStatus SetMatmulTilingMatmulReduceScatter(gert::TilingContext* context,
                                                       MatmulReduceScatterTilingData& tilingData,
                                                       mc2tiling::TilingArgs& args);
    ge::graphStatus GetReduceScatterFormulateTileCnt(const gert::TilingContext* ctx,
                                                     MatmulReduceScatterTilingData& tilingData,
                                                     mc2tiling::TilingArgs& args);
    bool IsDeterministic();
    virtual CutResult GetCutResult(MatmulReduceScatterTilingData& tilingData, mc2tiling::TilingArgs& args) = 0;
    virtual ge::graphStatus CheckValidRank(const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK,
                                           MatmulReduceScatterTilingData* tilingData, gert::TilingContext* context,
                                           uint32_t rankSize) = 0;
    virtual std::string GetRsConfig(MatmulReduceScatterTilingData& tilingData) = 0;
};
} // namespace optiling

#endif
