/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback.h"
#include "fallback/fallback_comm.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t EXPANDED_PERMUTED_ROWS_INDEX = 0;
static const size_t SKIP1_INDEX = 1;
static const size_t SKIP2_OPTIONAL_INDEX = 2;
static const size_t BIASE_INDEX = 3;
static const size_t SCALES_INDEX = 4;
static const size_t EXPANDED_SRC_TO_DST_ROW_INDEX = 5;
static const size_t EXPERT_FOR_SOURCE_ROW_INDEX = 6;
static const size_t INPUT_NUM = 7;

static graphStatus MoeFinalizeRoutingHostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_LOGD("aclnnFallback", "MoeFinalizeRouting fallback begin");

    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto expanded_permuted_rows = host_api_ctx->GetInputTensor(EXPANDED_PERMUTED_ROWS_INDEX);
    OP_CHECK_IF(expanded_permuted_rows == nullptr, OP_LOGE("aclnnfallback", "expanded_permuted_rows is null"),
                return GRAPH_FAILED);

    auto skip1 = host_api_ctx->GetInputTensor(SKIP1_INDEX);
    OP_CHECK_IF(skip1 == nullptr, OP_LOGE("aclnnfallback", "skip1 is null"), return GRAPH_FAILED);

    auto skip2_optional = host_api_ctx->GetOptionalInputTensor(SKIP2_OPTIONAL_INDEX);

    size_t offset = (host_api_ctx->GetComputeNodeInputNum() == INPUT_NUM) ? 0 : 1;
    auto bias = host_api_ctx->GetInputTensor(BIASE_INDEX - offset);
    OP_CHECK_IF(bias == nullptr, OP_LOGE("aclnnfallback", "bias is null"), return GRAPH_FAILED);

    auto scales = host_api_ctx->GetInputTensor(SCALES_INDEX - offset);
    OP_CHECK_IF(scales == nullptr, OP_LOGE("aclnnfallback", "scales is null"), return GRAPH_FAILED);

    auto expanded_src_to_dst_row = host_api_ctx->GetInputTensor(EXPANDED_SRC_TO_DST_ROW_INDEX - offset);
    OP_CHECK_IF(expanded_src_to_dst_row == nullptr, OP_LOGE("aclnnfallback", "expanded_src_to_dst_row is null"),
                return GRAPH_FAILED);

    auto expert_for_source_row = host_api_ctx->GetInputTensor(EXPERT_FOR_SOURCE_ROW_INDEX - offset);
    OP_CHECK_IF(expert_for_source_row == nullptr, OP_LOGE("aclnnfallback", "expert_for_source_row is null"),
                return GRAPH_FAILED);

    auto output = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(output == nullptr, OP_LOGE("aclnnfallback", "output is null"), return GRAPH_FAILED);

    // execute opapi
    auto api_ret = EXEC_OPAPI_CMD(aclnnMoeFinalizeRouting, expanded_permuted_rows, skip1, skip2_optional, bias, scales,
                                  expanded_src_to_dst_row, expert_for_source_row, output);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "api_ret faild:%u", api_ret),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MoeFinalizeRouting).OpExecuteFunc(MoeFinalizeRoutingHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
