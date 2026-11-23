/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include <string>
#include <vector>
#include "fallback/fallback.h"
#include "fallback/fallback_comm.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t EXPANDED_PERMUTED_ROWS_INDEX = 0;
static const size_t EXPANDED_SRC_TO_DST_ROW_INDEX = 1;
static const size_t SKIP1_INDEX = 2;
static const size_t SKIP2_OPTIONAL_INDEX = 3;
static const size_t BIASE_INDEX = 4;
static const size_t SCALES_INDEX = 5;
static const size_t EXPERT_FOR_SOURCE_ROW_INDEX = 6;
static const size_t X_INDEX = 7;
static const size_t ALPHA1_INDEX = 8;
static const size_t ALPHA2_INDEX = 9;
static const size_t V_INDEX = 10;
static const size_t EXPANDED_X_SCALE_INDEX = 11;

static std::vector<int64_t> GetExpertRange(const gert::ContinuousVector* ptr, bool isMx4)
{
    std::vector<int64_t> range = {-1, -1};
    if (ptr != nullptr) {
        const int64_t* data = reinterpret_cast<const int64_t*>(ptr->GetData());
        range.assign(data, data + ptr->GetSize());
    }
    // The operator definition uses [-1, -1] for an omitted range.
    if (isMx4 && range.size() == 2 && range[0] == -1 && range[1] == -1) {
        return {};
    }
    return range;
}

static graphStatus MoeFinalizeRoutingV2HostExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_LOGD("aclnnFallback", "MoeFinalizeRoutingV2 fallback begin");

    OP_CHECK_IF(host_api_ctx == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "host_api_ctx", "host_api_ctx is null"),
                return GRAPH_FAILED);

    auto expanded_permuted_rows = host_api_ctx->GetInputTensor(EXPANDED_PERMUTED_ROWS_INDEX);
    OP_CHECK_IF(expanded_permuted_rows == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "expanded_permuted_rows",
                                                         "expanded_permuted_rows is null"),
                return GRAPH_FAILED);

    auto expanded_src_to_dst_row = host_api_ctx->GetInputTensor(EXPANDED_SRC_TO_DST_ROW_INDEX);
    OP_CHECK_IF(expanded_src_to_dst_row == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "expanded_src_to_dst_row",
                                                         "expanded_src_to_dst_row is null"),
                return GRAPH_FAILED);

    auto skip1 = host_api_ctx->GetOptionalInputTensor(SKIP1_INDEX);

    auto skip2_optional = host_api_ctx->GetOptionalInputTensor(SKIP2_OPTIONAL_INDEX);

    auto bias = host_api_ctx->GetOptionalInputTensor(BIASE_INDEX);

    auto scales = host_api_ctx->GetOptionalInputTensor(SCALES_INDEX);

    auto expert_for_source_row = host_api_ctx->GetOptionalInputTensor(EXPERT_FOR_SOURCE_ROW_INDEX);

    auto x = host_api_ctx->GetOptionalInputTensor(X_INDEX);

    auto alpha1 = host_api_ctx->GetOptionalInputTensor(ALPHA1_INDEX);

    auto alpha2 = host_api_ctx->GetOptionalInputTensor(ALPHA2_INDEX);

    auto v = host_api_ctx->GetOptionalInputTensor(V_INDEX);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "attrs", "attrs is null"),
                return GRAPH_FAILED);
    const int64_t* mode = attrs->GetAttrPointer<int64_t>(0);
    const gert::ContinuousVector* zero_expert_range_ptr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const gert::ContinuousVector* copy_expert_range_ptr = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    const gert::ContinuousVector* constant_expert_range_ptr = attrs->GetAttrPointer<gert::ContinuousVector>(3);
    const int64_t* k = attrs->GetAttrPointer<int64_t>(4);

    const bool isMx4 = expanded_permuted_rows->GetDataType() == ge::DT_FLOAT4_E2M1;
    const auto zeroExpertRange = GetExpertRange(zero_expert_range_ptr, isMx4);
    const auto copyExpertRange = GetExpertRange(copy_expert_range_ptr, isMx4);
    const auto constantExpertRange = GetExpertRange(constant_expert_range_ptr, isMx4);

    auto output = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(output == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "output", "output is null"),
                return GRAPH_FAILED);

    // EXEC_OPAPI_CMD converts and releases the expert range arrays.
    const auto expandedXScale = isMx4 ? host_api_ctx->GetOptionalInputTensor(EXPANDED_X_SCALE_INDEX) : nullptr;
    const auto api_ret =
        isMx4 ?
            EXEC_OPAPI_CMD(aclnnMoeFinalizeRoutingV5, expanded_permuted_rows, expanded_src_to_dst_row, skip1,
                           skip2_optional, bias, scales, expert_for_source_row, x, alpha1, alpha2, v, expandedXScale,
                           *mode, zeroExpertRange, copyExpertRange, constantExpertRange, *k, output) :
            EXEC_OPAPI_CMD(aclnnMoeFinalizeRoutingV4, expanded_permuted_rows, expanded_src_to_dst_row, skip1,
                           skip2_optional, bias, scales, expert_for_source_row, x, alpha1, alpha2, v, *mode,
                           zeroExpertRange, copyExpertRange, constantExpertRange, *k, output);
    OP_CHECK_IF(
        api_ret != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(host_api_ctx->GetNodeName(), "aclnn status",
                                              std::to_string(api_ret).c_str(), "MoeFinalizeRoutingV2 fallback failed."),
        return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MoeFinalizeRoutingV2).OpExecuteFunc(MoeFinalizeRoutingV2HostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
