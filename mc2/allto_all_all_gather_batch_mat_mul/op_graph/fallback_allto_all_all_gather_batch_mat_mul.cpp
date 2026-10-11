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
 * \file fallback_allto_all_all_gather_batch_mat_mul.cpp
 * \brief fallback function of op allto_all_all_gather_batch_mat_mul
 */

#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"

namespace fallback {
static const char* fallbackInfo = "AlltoAllAllGatherBmmFallback";

static ge::graphStatus AlltoAllAllGatherBmmExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D("Start to fallback for all to all allgather bmm.");
    OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(fallbackInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);

    const auto x = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    OPS_ERR_IF(x == nullptr, OPS_LOG_E(fallbackInfo, "x is null"), return ge::GRAPH_FAILED);

    const auto weight = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    OPS_ERR_IF(weight == nullptr, OPS_LOG_E(fallbackInfo, "weight is null"), return ge::GRAPH_FAILED);

    const auto bias = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));

    const auto y1 = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y1));
    OPS_ERR_IF(y1 == nullptr, OPS_LOG_E(fallbackInfo, "y1 is null"), return ge::GRAPH_FAILED);

    // output 不存在可选，只是在语义上可选，就算是空 shape 也不会返回 nullptr
    // 可以 infershape 时候指定为空 shape --- 一个维度为 0 或者根据 flag 填充 nullptr 进去
    auto y2 = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y2));
    OPS_ERR_IF(y2 == nullptr, OPS_LOG_E(fallbackInfo, "y2 is null"), return ge::GRAPH_FAILED);

    auto y3 = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y3));
    OPS_ERR_IF(y3 == nullptr, OPS_LOG_E(fallbackInfo, "y3 is null"), return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(fallbackInfo, "attrs is null"), return ge::GRAPH_FAILED);

    const char* groupEp = attrs->GetStr(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_GROUP_EP));
    const char* groupTp = attrs->GetStr(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_GROUP_TP));
    const int64_t* epWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_EP_WORLD_SIZE));
    const int64_t* tpWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_TP_WORLD_SIZE));
    const int64_t* xShardType = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_X_SHARD_TYPE));
    const bool* isTransW = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_IS_TRANS_W));
    const int64_t* actType = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_ACT_TYPE));
    // 读取 flag 判断是否需要 optional output
    const bool* outputY2Flag = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_OUTPUT_Y2_FLAG));
    const bool* outputY3Flag = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_OUTPUT_Y3_FLAG));
    if (!(*outputY2Flag)) {
        OPS_LOG_D(fallbackInfo, "do not output y2");
        y2 = nullptr;
    }
    if (!(*outputY3Flag)) {
        OPS_LOG_D(fallbackInfo, "do not output y3");
        y3 = nullptr;
    }

    auto weightAcl = ConvertMmType(weight, *isTransW); // 手动转换为 aclTensor，并根据转置信息处理 stride
    OPS_ERR_IF(weightAcl == nullptr, OPS_LOG_E(fallbackInfo, "weightAcl is null"), return ge::GRAPH_FAILED);

    const auto apiRet = EXEC_OPAPI_CMD(aclnnAlltoAllAllGatherBatchMatMul, x, weightAcl, bias, groupEp, groupTp,
                                       *epWorldSize, *tpWorldSize, *xShardType, *actType, y1, y2, y3);
    OPS_ERR_IF(apiRet != ge::GRAPH_SUCCESS, OPS_LOG_E(fallbackInfo, "Aclnn api error code %u", apiRet),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AlltoAllAllGatherBatchMatMul).OpExecuteFunc(AlltoAllAllGatherBmmExecuteFunc);
} // namespace fallback
