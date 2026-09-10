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

static std::vector<int64_t> GetExpertRange(const gert::ContinuousVector *ptr)
{
    std::vector<int64_t> range = {-1, -1};
    if (ptr != nullptr) {
        const int64_t *data = reinterpret_cast<const int64_t *>(ptr->GetData());
        range.assign(data, data + ptr->GetSize());
    }
    return range;
}

using CreateIntArrayFunc = aclIntArray *(*)(const int64_t *, uint64_t);
using DestroyIntArrayFunc = int (*)(const aclIntArray *);

static aclIntArray *CreateExpertRange(const std::vector<int64_t> &range)
{
    if (range.empty()) {
        return nullptr;
    }
    static const auto aclCreateIntArray = reinterpret_cast<CreateIntArrayFunc>(GetOpApiFuncAddr("aclCreateIntArray"));
    OP_CHECK_IF(aclCreateIntArray == nullptr, OP_LOGE("aclnnfallback", "aclCreateIntArray is null"), return nullptr);
    return aclCreateIntArray(range.data(), range.size());
}

class ExpertRangeGuard {
public:
    explicit ExpertRangeGuard(const std::vector<int64_t> &range)
        : range_(CreateExpertRange(range))
    {}

    ~ExpertRangeGuard()
    {
        if (range_ == nullptr) {
            return;
        }
        static const auto aclDestroyIntArray =
            reinterpret_cast<DestroyIntArrayFunc>(GetOpApiFuncAddr("aclDestroyIntArray"));
        OP_CHECK_IF(aclDestroyIntArray == nullptr, OP_LOGE("aclnnfallback", "aclDestroyIntArray is null"), return);
        aclDestroyIntArray(range_);
    }

    aclIntArray *Get() const
    {
        return range_;
    }

private:
    aclIntArray *range_;
};

static graphStatus MoeFinalizeRoutingV2HostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_LOGD("aclnnFallback", "MoeFinalizeRoutingV2 fallback begin");

    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto expanded_permuted_rows = host_api_ctx->GetInputTensor(EXPANDED_PERMUTED_ROWS_INDEX);
    OP_CHECK_IF(expanded_permuted_rows == nullptr, OP_LOGE("aclnnfallback", "expanded_permuted_rows is null"),
                return GRAPH_FAILED);

    auto expanded_src_to_dst_row = host_api_ctx->GetInputTensor(EXPANDED_SRC_TO_DST_ROW_INDEX);
    OP_CHECK_IF(expanded_src_to_dst_row == nullptr, OP_LOGE("aclnnfallback", "expanded_src_to_dst_row is null"),
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
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    const int64_t *mode = attrs->GetAttrPointer<int64_t>(0);
    const gert::ContinuousVector *zero_expert_range_ptr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const gert::ContinuousVector *copy_expert_range_ptr = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    const gert::ContinuousVector *constant_expert_range_ptr = attrs->GetAttrPointer<gert::ContinuousVector>(3);
    const int64_t *k = attrs->GetAttrPointer<int64_t>(4);

    std::vector<int64_t> zeroExpertRangeVec = GetExpertRange(zero_expert_range_ptr);
    std::vector<int64_t> copyExpertRangeVec = GetExpertRange(copy_expert_range_ptr);
    std::vector<int64_t> constantExpertRangeVec = GetExpertRange(constant_expert_range_ptr);
    ExpertRangeGuard zeroExpertRangeAcl(zeroExpertRangeVec);
    ExpertRangeGuard copyExpertRangeAcl(copyExpertRangeVec);
    ExpertRangeGuard constantExpertRangeAcl(constantExpertRangeVec);
    aclIntArray *zeroExpertRange = zeroExpertRangeAcl.Get();
    aclIntArray *copyExpertRange = copyExpertRangeAcl.Get();
    aclIntArray *constantExpertRange = constantExpertRangeAcl.Get();

    auto output = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(output == nullptr, OP_LOGE("aclnnfallback", "output is null"), return GRAPH_FAILED);

    // execute opapi
    auto api_ret = EXEC_OPAPI_CMD(aclnnMoeFinalizeRoutingV4, expanded_permuted_rows, expanded_src_to_dst_row, skip1,
                                  skip2_optional, bias, scales, expert_for_source_row, x, alpha1, alpha2, v, *mode,
                                  zeroExpertRange, copyExpertRange, constantExpertRange, *k, output);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "api_ret faild:%u", api_ret),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MoeFinalizeRoutingV2).OpExecuteFunc(MoeFinalizeRoutingV2HostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
