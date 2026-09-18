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
 * \file all_gather_matmul_tiling_a2a3.cpp
 * \brief
 */
#include "all_gather_formulaic_tiling_a2a3.h"
#include "all_gather_matmul_tiling_a2a3.h"

namespace optiling {

ge::graphStatus AllGatherMatmulTilingA2A3::CheckValidRank(Mc2Tiling::AllGatherMatmulTilingData *tilingData,
                                                          const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK,
                                                          gert::TilingContext *context, uint32_t rankSize)
{
    // distinguish between 910A2 and 910A3
    auto it = std::find(VALID_RANK.at(tilingData->socParam.isA3).begin(),
                        VALID_RANK.at(tilingData->socParam.isA3).end(), rankSize);
    OP_TILING_CHECK(it == VALID_RANK.at(tilingData->socParam.isA3).end(),
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "world_size", std::to_string(rankSize).c_str(),
                                              "valid rank value"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void AllGatherMatmulTilingA2A3::SetSocParam(Mc2Tiling::AllGatherMatmulTilingData *tilingData, const char *group)
{
    auto commSets = mc2tiling::Mc2TilingUtils::GetCommSets(group);
    tilingData->socParam.isA3 = (commSets == mc2tiling::COMM_MESH) ? 0 : 1;
    tilingData->socParam.isStep = 0U;
    tilingData->socParam.isND2NZ = 1U;
}

std::string AllGatherMatmulTilingA2A3::GetAlgConfig(Mc2Tiling::AllGatherMatmulTilingData *tilingData,
                                                    [[maybe_unused]] const mc2tiling::TilingArgs &args)
{
    return (tilingData->socParam.isA3 == 0) ? "AllGather=level0:fullmesh" : "AllGather=level0:doublering";
}

CutResult AllGatherMatmulTilingA2A3::GetCutResult(Mc2Tiling::AllGatherMatmulTilingData &tilingData,
                                                  mc2tiling::TilingArgs &args)
{
    SocVersion_2201 socVersion_2201 =
        (tilingData.socParam.isA3 == 0) ? SocVersion_2201::SOC910_B : SocVersion_2201::SOC910_93;
    AllGatherPlusMMA2A3 tileFormulate(args, args.rankDim, KernelType::ALL_GATHER, socVersion_2201);
    tileFormulate.GetTiling();
    CutResult mCutGather = tileFormulate.tilingM_.cutRes;
    return mCutGather;
}

static ge::graphStatus AllGatherMatmulTilingFuncA2A3(gert::TilingContext *context)
{
    AllGatherMatmulTilingA2A3 impl;
    return impl.AllGatherMatmulTilingFunc(context);
}

struct AllGatherMatmulCompileInfo {};
static ge::graphStatus TilingParseForAllGatherMatmul([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(AllGatherMatmul)
    .Tiling(AllGatherMatmulTilingFuncA2A3)
    .TilingParse<AllGatherMatmulCompileInfo>(TilingParseForAllGatherMatmul);

} // namespace optiling
