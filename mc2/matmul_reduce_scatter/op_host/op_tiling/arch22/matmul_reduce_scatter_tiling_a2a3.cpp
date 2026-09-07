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
 * \file matmul_reduce_scatter_tiling_a2a3.cpp
 * \brief
 */
#include "matmul_reduce_scatter_tiling_a2a3.h"
#include "vector"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_hcom_topo_info.h"
#include "util/math_util.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "matmul_reduce_scatter_v2/op_host/op_tiling/reduce_scatter_formulaic_tiling.h"
#include "../../../op_kernel/matmul_reduce_scatter_tiling_key.h"
#include "../../../op_kernel/matmul_reduce_scatter_tiling.h"

namespace optiling {
CutResult MatmulReduceScatterTilingFuncA2A3::GetCutResult(MatmulReduceScatterTilingData &tilingData,
                                                          mc2tiling::TilingArgs &args)
{
    SocVersion inputSocVersion = (tilingData.socParam.isA3 == 0) ? SocVersion::SOC910_B : SocVersion::SOC910_93;
    bool commDeterministic = (inputSocVersion == SocVersion::SOC910_B) ? IsDeterministic() : false;
    MMPlusReduceScatter scatterTilingHccl(args, args.rankDim, KernelType::REDUCE_SCATTER, inputSocVersion,
                                          commDeterministic);
    scatterTilingHccl.GetTiling();

    return scatterTilingHccl.tilingM_.cutRes;
}

ge::graphStatus MatmulReduceScatterTilingFuncA2A3::CheckValidRank(
    const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK, MatmulReduceScatterTilingData *tilingData,
    gert::TilingContext *context, uint32_t rankSize)
{
    // distinguish between 910A2 and 910A3
    auto it = std::find(VALID_RANK.at(tilingData->socParam.isA3).begin(),
                        VALID_RANK.at(tilingData->socParam.isA3).end(), rankSize);
    OP_TILING_CHECK(
        it == VALID_RANK.at(tilingData->socParam.isA3).end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "world_size", std::to_string(rankSize).c_str(),
                                              "The value of world_size is illegal"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

std::string MatmulReduceScatterTilingFuncA2A3::GetRsConfig(MatmulReduceScatterTilingData &tilingData)
{
    std::string rsConfig =
        (tilingData.socParam.isA3 == 0) ? "ReduceScatter=level0:fullmesh" : "ReduceScatter=level0:doublering";
    return rsConfig;
}

static ge::graphStatus MatmulReduceScatterTilingImplFuncA2A3(gert::TilingContext *context)
{
    MatmulReduceScatterTilingFuncA2A3 impl;
    return impl.MatmulReduceScatterTilingFunc(context);
}

struct MatmulReduceScatterCompileInfo {};
static ge::graphStatus TilingParseForMatmulReduceScatter(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulReduceScatter)
    .Tiling(MatmulReduceScatterTilingImplFuncA2A3)
    .TilingParse<MatmulReduceScatterCompileInfo>(TilingParseForMatmulReduceScatter);
} // namespace optiling
