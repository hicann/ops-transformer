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
 * \file fallback_quant_reduce_scatter.cpp
 * \brief 动态shape图回调aclnn
 */
#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"

namespace fallback {

const char *QuantReduceScatterInfo = "QuantReduceScatterFallback";

// input
constexpr uint32_t CONTEXT_IDX = 0;
constexpr uint32_t X_IDX = 1;
constexpr uint32_t SCALES_IDX = 2;
// output
constexpr uint32_t OUTPUT_IDX = 0;
// attr
constexpr uint32_t HCCL_BUFFER_SIZE_IDX = 0;
constexpr uint32_t REDUCE_OP_IDX = 1;
constexpr uint32_t WORLD_SIZE_IDX = 3;

static ge::graphStatus QuantReduceScatterExecuteFunc(gert::OpExecuteContext *host_api_ctx)
{
    OPS_LOG_D(QuantReduceScatterInfo, "Start to fallback for quant_reduce_scatter.");
    OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "host_api_ctx is null"),
               return ge::GRAPH_FAILED);

    // 校验tensor
    const auto context = host_api_ctx->GetInputTensor(static_cast<size_t>(CONTEXT_IDX));
    OPS_ERR_IF(context == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "context is null"), return ge::GRAPH_FAILED);
    const auto x = host_api_ctx->GetInputTensor(static_cast<size_t>(X_IDX));
    OPS_ERR_IF(x == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "x is null"), return ge::GRAPH_FAILED);
    const auto scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(SCALES_IDX));
    OPS_ERR_IF(scales == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "scales is null"), return ge::GRAPH_FAILED);

    const auto output = host_api_ctx->GetOutputTensor(static_cast<size_t>(OUTPUT_IDX));
    OPS_ERR_IF(output == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "output is null"), return ge::GRAPH_FAILED);

    // 校验attrs
    const auto attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "attrs is null"), return ge::GRAPH_FAILED);
    const int64_t *hccl_buffer_size = attrs->GetInt(static_cast<size_t>(HCCL_BUFFER_SIZE_IDX));
    OPS_ERR_IF(hccl_buffer_size == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "hccl_buffer_size is null"),
               return ge::GRAPH_FAILED);
    const char *reduce_op = attrs->GetStr(static_cast<size_t>(REDUCE_OP_IDX));
    OPS_ERR_IF(reduce_op == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "reduce_op is null"), return ge::GRAPH_FAILED);
    const int64_t *world_size = attrs->GetInt(static_cast<size_t>(WORLD_SIZE_IDX));
    OPS_ERR_IF(world_size == nullptr, OPS_LOG_E(QuantReduceScatterInfo, "world_size is null"), return ge::GRAPH_FAILED);

    // 执行回调
    const auto ret =
        EXEC_OPAPI_CMD(aclnnQuantReduceScatter, context, x, scales, *hccl_buffer_size, *world_size, reduce_op, output);
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS, OPS_LOG_E(QuantReduceScatterInfo, "Aclnn api error code %d", ret),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantReduceScatter).OpExecuteFunc(QuantReduceScatterExecuteFunc);

} // namespace fallback
