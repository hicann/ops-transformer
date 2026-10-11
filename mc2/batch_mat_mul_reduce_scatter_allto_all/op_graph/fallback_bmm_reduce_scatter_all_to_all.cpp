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
 * \file fallback_bmm_reduce_scatter_all_to_all.cpp
 * \brief fallback function of op bmm_reduce_scatter_all_to_all
 */

#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"

namespace fallback {
static const char* fallbackInfo = "BmmReduceScatterAlltoAllFallback";

static ge::graphStatus BmmReduceScatterAlltoAllExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D("Start to fallback for all to all allgather bmm.");
    OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(fallbackInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);

    const auto x = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    OPS_ERR_IF(x == nullptr, OPS_LOG_E(fallbackInfo, "x is null"), return ge::GRAPH_FAILED);

    const auto weight = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    OPS_ERR_IF(weight == nullptr, OPS_LOG_E(fallbackInfo, "weight is null"), return ge::GRAPH_FAILED);

    const auto bias = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));

    const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::BmmReduceScatterAlltoAllOutIdx::K_Y));
    OPS_ERR_IF(y == nullptr, OPS_LOG_E(fallbackInfo, "y is null"), return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(fallbackInfo, "attrs is null"), return ge::GRAPH_FAILED);

    const char* groupEp = attrs->GetStr(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_GROUP_EP));
    const char* groupTp = attrs->GetStr(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_GROUP_TP));
    const int64_t* epSize = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_EP_WORLD_SIZE));
    const int64_t* tpSize = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_TP_WORLD_SIZE));
    const int64_t* yShard = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_Y_SHARD_TYPE));
    const bool* isTransW = attrs->GetBool(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_IS_TRANS_W));

    auto weightAcl = ConvertMmType(weight, *isTransW); // 手动转换为 aclTensor，并根据转置信息处理 stride
    OPS_ERR_IF(weightAcl == nullptr, OPS_LOG_E(fallbackInfo, "weightAcl is null"), return ge::GRAPH_FAILED);

    const auto apiRet = EXEC_OPAPI_CMD(aclnnBatchMatMulReduceScatterAlltoAll, x, weightAcl, bias, groupEp, groupTp,
                                       *epSize, *tpSize, *yShard, y);
    OPS_ERR_IF(apiRet != ge::GRAPH_SUCCESS, OPS_LOG_E(fallbackInfo, "Aclnn api error code %u", apiRet),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BatchMatMulReduceScatterAlltoAll).OpExecuteFunc(BmmReduceScatterAlltoAllExecuteFunc);
} // namespace fallback
