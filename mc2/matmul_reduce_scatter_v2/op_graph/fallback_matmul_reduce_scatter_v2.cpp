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
 * \file fallback_matmul_reduce_scatter_v2.cpp
 * \brief
 */

#include <cstring>
#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"

namespace fallback {
const char *reduceScatterV2Info = "MmReduceScatterV2Fallback";
constexpr char GRAPH_CCU_MODE[] = "ccu_graph";

static ge::graphStatus MatmulReduceScatterV2ExecuteFunc(gert::OpExecuteContext *host_api_ctx)
{
    OPS_LOG_D(reduceScatterV2Info, "Start to fallback for matmul reducescatter v2.");
    OPS_CHECK(host_api_ctx == nullptr, OPS_LOG_E(reduceScatterV2Info, "host_api_ctx is null"), return ge::GRAPH_FAILED);

    const auto x1 = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MC2V2InputIdx::K_X1));
    OPS_CHECK(x1 == nullptr, OPS_LOG_E(reduceScatterV2Info, "x1 is null"), return ge::GRAPH_FAILED);

    const auto x2 = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MC2V2InputIdx::K_X2));
    OPS_CHECK(x2 == nullptr, OPS_LOG_E(reduceScatterV2Info, "x2 is null"), return ge::GRAPH_FAILED);

    const auto bias = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2V2InputIdx::K_BIAS));

    const auto x1Scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2V2InputIdx::K_X1SCALE));

    const auto x2Scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2V2InputIdx::K_X2SCALE));

    const auto quantScale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2V2InputIdx::K_QUNATSCALE));

    const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MC2ReduceScatterV2OutputIdx::K_Y));
    OPS_CHECK(y == nullptr, OPS_LOG_E(reduceScatterV2Info, "y is null"), return ge::GRAPH_FAILED);

    const auto amaxOut = nullptr;

    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OPS_LOG_E(reduceScatterV2Info, "attrs is null"), return ge::GRAPH_FAILED);

    const int64_t *blockSizePtr = attrs->GetInt(static_cast<size_t>(ops::MmReduceScatterV2AttrIdx::K_BLOCK_SIZE));
    const int64_t blockSize = (blockSizePtr != nullptr ? *blockSizePtr : 0);

    const char *group = attrs->GetStr(static_cast<size_t>(ops::MmReduceScatterV2AttrIdx::K_GROUP));
    OPS_CHECK(group == nullptr, OPS_LOG_E(reduceScatterV2Info, "group is null"), return ge::GRAPH_FAILED);

    const char *op = attrs->GetStr(static_cast<size_t>(ops::MmReduceScatterV2AttrIdx::K_OP));
    OPS_CHECK(op == nullptr, OPS_LOG_E(reduceScatterV2Info, "reduceOp is null"), return ge::GRAPH_FAILED);

    const bool *transX1Ptr = attrs->GetBool(static_cast<size_t>(ops::MmReduceScatterV2AttrIdx::K_TRANS_X1));
    const bool transX1 = (transX1Ptr != nullptr ? *transX1Ptr : false);
    auto x1Acl = ConvertMmType(x1, transX1);
    OPS_CHECK(x1Acl == nullptr, OPS_LOG_E(reduceScatterV2Info, "x1Acl is null"), return ge::GRAPH_FAILED);

    const bool *transX2Ptr = attrs->GetBool(static_cast<size_t>(ops::MmReduceScatterV2AttrIdx::K_TRANS_X2));
    const bool transX2 = (transX2Ptr != nullptr ? *transX2Ptr : false);
    auto x2Acl = ConvertMmType(x2, transX2);
    OPS_CHECK(x2Acl == nullptr, OPS_LOG_E(reduceScatterV2Info, "x2Acl is null"), return ge::GRAPH_FAILED);

    const int64_t *commTurnPtr = attrs->GetInt(static_cast<size_t>(ops::MmReduceScatterV2AttrIdx::K_COMM_TURN));
    const int64_t commTurn = (commTurnPtr != nullptr ? *commTurnPtr : 0);
    const int64_t streamMode = 1; // STOP_ON_FAILURE

    const int64_t *groupSizePtr = attrs->GetInt(static_cast<size_t>(ops::MmReduceScatterV2AttrIdx::K_GROUP_SIZE));
    const int64_t groupSize = (groupSizePtr != nullptr ? *groupSizePtr : 0);
    const char *commMode = attrs->GetStr(static_cast<size_t>(ops::MmReduceScatterV2AttrIdx::K_COMM_MODE));
    OPS_CHECK(commMode == nullptr, OPS_LOG_E(reduceScatterV2Info, "commMode is null"), return ge::GRAPH_FAILED);

    // Do not let graph fallback be identified as a direct ACLNN peer-only call.
    const char *graphCommMode = (std::strcmp(commMode, "ccu") == 0) ? GRAPH_CCU_MODE : commMode;
    const auto apiRet =
        EXEC_OPAPI_CMD(aclnnMatmulReduceScatterV2, x1Acl, x2Acl, bias, x1Scale, x2Scale, quantScale, blockSize, group,
                       op, commTurn, streamMode, groupSize, graphCommMode, y, amaxOut);
    OPS_CHECK(apiRet != ge::GRAPH_SUCCESS, OPS_LOG_E(reduceScatterV2Info, "Aclnn api error code %d", apiRet),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MatmulReduceScatterV2).OpExecuteFunc(MatmulReduceScatterV2ExecuteFunc);
} // namespace fallback
