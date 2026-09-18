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
 * \file all_gather_matmul_tiling_a5.cpp
 * \brief
 */
#include "all_gather_formulaic_tiling_a5.h"
#include "all_gather_matmul_tiling_a5.h"
#include "all_gather_matmul_v2/op_host/op_tiling/arch35/all_gather_comm_algo_table.h"
#include "mc2_comm_algo_selector.h"
#include "mc2_comm_utils.h"

namespace optiling {

ge::graphStatus AllGatherMatmulTilingA5::CheckValidRank(Mc2Tiling::AllGatherMatmulTilingData *tilingData,
                                                        const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK,
                                                        gert::TilingContext *context, uint32_t rankSize)
{
    auto it = std::find(VALID_RANK.at(0).begin(), VALID_RANK.at(0).end(), rankSize);
    OP_TILING_CHECK(it == VALID_RANK.at(0).end(),
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "world_size", std::to_string(rankSize).c_str(),
                                              "valid rank value"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void AllGatherMatmulTilingA5::SetSocParam(Mc2Tiling::AllGatherMatmulTilingData *tilingData,
                                          [[maybe_unused]] const char *group)
{
    tilingData->socParam.isA3 = 0U;
    tilingData->socParam.isStep = 0U;
    tilingData->socParam.isND2NZ = 1U;
}

std::string AllGatherMatmulTilingA5::GetAlgConfig(Mc2Tiling::AllGatherMatmulTilingData *tilingData,
                                                  const mc2tiling::TilingArgs &args)
{
    uint32_t algoCount = 0;
    const Mc2Hcom::CommAlgoEntry *algoEntries = Mc2Tiling::GetAllGatherCommAlgoTable(algoCount);
    // 单轮通信量 = 每轮切分的 tileM × K × dtypeSize，
    // MCSpliteM 各分支结束时 args.mValue 即单轮 tileM，覆盖 formulate/enableSplitK/commTurn 全部路径；
    // 用于选择器按表项 [minBytes, maxBytes] 区间过滤，算法表为空时回退默认算法 ALLGATHER_DEFAULT_ALGO_NAME
    uint64_t commDataBytes = args.mValue * args.kValue * args.inputDtypeSize;
    // V1 tiling 侧无 comm_engine 概念，arch35 仅存在 AICPU 引擎路径（op_api 固定以 ai_cpu 转发），故固定传 AICPU
    std::string algConfig = Mc2Hcom::Mc2CommAlgoSelector::SelectAlgoName(
        "AllGatherMatmul", group_, Mc2Comm::ENGINE_AICPU, commDataBytes, tilingData->param.rankDim, algoEntries,
        algoCount, Mc2Tiling::ALLGATHER_DEFAULT_ALGO_NAME);
    OP_LOGD("AllGatherMatmul", "GetAlgConfig selected algConfig=%s, commDataBytes=%llu, rankDim=%u", algConfig.c_str(),
            commDataBytes, tilingData->param.rankDim);
    return algConfig;
}

CutResult AllGatherMatmulTilingA5::GetCutResult(Mc2Tiling::AllGatherMatmulTilingData &tilingData,
                                                mc2tiling::TilingArgs &args)
{
    AllGatherPlusMMA5 tileFormulate(args, args.rankDim, KernelType::ALL_GATHER);
    tileFormulate.GetTiling();
    CutResult mCutGather = tileFormulate.tilingM_.cutRes;
    return mCutGather;
}

static ge::graphStatus AllGatherMatmulTilingFuncA5(gert::TilingContext *context)
{
    AllGatherMatmulTilingA5 impl;
    return impl.AllGatherMatmulTilingFunc(context);
}

struct AllGatherMatmulCompileInfo {};
static ge::graphStatus TilingParseForAllGatherMatmul([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(AllGatherMatmul)
    .Tiling(AllGatherMatmulTilingFuncA5)
    .TilingParse<AllGatherMatmulCompileInfo>(TilingParseForAllGatherMatmul);

} // namespace optiling
