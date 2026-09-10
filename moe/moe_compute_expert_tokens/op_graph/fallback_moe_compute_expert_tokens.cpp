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
static const size_t SORTED_EXPERTS_INDEXQ = 0;

static graphStatus MoeComputeHostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_LOGD("aclnnFallback", "MoeComputeExpertTokens fallback begin");

    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto sortedExperts = host_api_ctx->GetInputTensor(SORTED_EXPERTS_INDEXQ);
    OP_CHECK_IF(sortedExperts == nullptr, OP_LOGE("aclnnfallback", "sortedExperts is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    const int32_t *num_experts = attrs->GetAttrPointer<int32_t>(0);
    OP_CHECK_IF(*num_experts < 1, OP_LOGE("aclnnfallback", "num_experts can not be smaller than 1"),
                return GRAPH_FAILED);

    auto output = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(output == nullptr, OP_LOGE("aclnnfallback", "output is null"), return GRAPH_FAILED);

    // execute opapi
    auto api_ret = EXEC_OPAPI_CMD(aclnnMoeComputeExpertTokens, sortedExperts, *num_experts, output);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "api_ret faild:%u", api_ret), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MoeComputeExpertTokens).OpExecuteFunc(MoeComputeHostExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
