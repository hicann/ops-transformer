/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fallback_moe_update_expert.cpp
 * \brief fallback function of op MoeUpdateExpert
 */

#include "fallback/fallback.h"
#include "mc2_log_compat.h"

namespace fallback {
static constexpr size_t IDX_IN_EXPERT_IDS = 0;
static constexpr size_t IDX_IN_EPLB_TABLE = 1;
static constexpr size_t IDX_IN_EXPERT_SCALES = 2;
static constexpr size_t IDX_IN_PRUNING_THRESHOLD = 3;
static constexpr size_t IDX_IN_ACTIVE_MASK = 4;

static constexpr size_t IDX_ATTR_LOCAL_RANK_ID = 0;
static constexpr size_t IDX_ATTR_WORLD_SIZE = 1;
static constexpr size_t IDX_ATTR_BALANCE_MODE = 2;

static constexpr size_t IDX_OUT_EXPERT_IDS = 0;
static constexpr size_t IDX_OUT_ACTIVE_MASK = 1;
static ge::graphStatus MoeUpdateExpertExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OP_LOGD("Start MoeUpdateExpertFallback.");
    OPS_CHECK(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "host_api_ctx"),
              return ge::GRAPH_FAILED);

    const auto expertIds = host_api_ctx->GetInputTensor(static_cast<size_t>(IDX_IN_EXPERT_IDS));
    OPS_CHECK(expertIds == nullptr, OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "expertIds"),
              return ge::GRAPH_FAILED);

    const auto eplbTable = host_api_ctx->GetInputTensor(static_cast<size_t>(IDX_IN_EPLB_TABLE));
    OPS_CHECK(eplbTable == nullptr, OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "eplbTable"),
              return ge::GRAPH_FAILED);

    const auto expertScales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(IDX_IN_EXPERT_SCALES));
    const auto pruningThreshold = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(IDX_IN_PRUNING_THRESHOLD));
    const auto activeMask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(IDX_IN_ACTIVE_MASK));

    const auto balancedExpertIds = host_api_ctx->GetOutputTensor(static_cast<size_t>(IDX_OUT_EXPERT_IDS));
    OPS_CHECK(balancedExpertIds == nullptr, OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "balancedExpertIds"),
              return ge::GRAPH_FAILED);

    const auto balancedActiveMask = host_api_ctx->GetOutputTensor(static_cast<size_t>(IDX_OUT_ACTIVE_MASK));
    OPS_CHECK(balancedActiveMask == nullptr,
              OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "balancedActiveMask"), return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "attrs"),
              return ge::GRAPH_FAILED);

    const auto localRankId = attrs->GetInt(static_cast<size_t>(IDX_ATTR_LOCAL_RANK_ID));
    OPS_CHECK(localRankId == nullptr, OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "localRankId"),
              return ge::GRAPH_FAILED);

    const auto worldSize = attrs->GetInt(static_cast<size_t>(IDX_ATTR_WORLD_SIZE));
    OPS_CHECK(worldSize == nullptr, OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "worldSize"),
              return ge::GRAPH_FAILED);

    const auto balanceMode = attrs->GetInt(static_cast<size_t>(IDX_ATTR_BALANCE_MODE));
    OPS_CHECK(balanceMode == nullptr, OP_LOGE_WITH_INVALID_INPUT("MoeUpdateExpertFallback", "balanceMode"),
              return ge::GRAPH_FAILED);

    const auto apiRet =
        EXEC_OPAPI_CMD(aclnnMoeUpdateExpert, expertIds, eplbTable, expertScales, pruningThreshold, activeMask,
                       *localRankId, *worldSize, *balanceMode, balancedExpertIds, balancedActiveMask);

    OPS_CHECK(apiRet != ge::GRAPH_SUCCESS, OP_LOGE("MoeUpdateExpertFallback", "Aclnn api error code %u", apiRet),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MoeUpdateExpert).OpExecuteFunc(MoeUpdateExpertExecuteFunc);
} // namespace fallback
