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
 * \file aclnn_moe_token_unpermute.cpp
 * \brief
 */
#include "aclnn_moe_token_unpermute.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_token_unpermute.h" // 该文件为自动生成，在build/autogen/inner路径下
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe/moe_finalize_routing_v2/op_host/op_api/moe_finalize_routing_v2.h"

using namespace op;

namespace MoeTokenUnpermuteCheck {

static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_X = {DataType::DT_FLOAT16, DataType::DT_BF16,
                                                                             DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *permutedTokens, const aclTensor *sortedIndices, const aclTensor *out)
{
    OP_CHECK_NULL(permutedTokens, return false);
    OP_CHECK_NULL(sortedIndices, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *permutedTokens, const aclTensor *sortedIndices,
                                   const aclTensor *probsOptional, const aclTensor *out)
{
    if (permutedTokens != nullptr && permutedTokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permutedTokens, MOE_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (sortedIndices != nullptr && sortedIndices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndices, MOE_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (probsOptional != nullptr && probsOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsOptional, MOE_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (out != nullptr && out->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(out, MOE_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(permutedTokens, out, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *permutedTokens, const aclTensor *sortedIndices,
                               const aclTensor *probsOptional, const aclTensor *out)
{
    CHECK_RET(CheckNotNull(permutedTokens, sortedIndices, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(permutedTokens, sortedIndices, probsOptional, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenUnpermuteCheck

#ifdef __cplusplus
extern "C" {
#endif

constexpr int64_t READ_INDEX_BY_ROW = 2;

aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(const aclTensor *permutedTokens, const aclTensor *sortedIndices,
                                                   const aclTensor *probsOptional, bool paddedMode,
                                                   const aclIntArray *restoreShapeOptional, aclTensor *out,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenUnpermute,
                   DFX_IN(permutedTokens, sortedIndices, probsOptional, paddedMode, restoreShapeOptional),
                   DFX_OUT(out));

    bool useMoeFinalizeRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2) {
        return aclnnInnerMoeTokenUnpermuteGetWorkspaceSize(permutedTokens, sortedIndices, probsOptional, paddedMode,
                                                           restoreShapeOptional, out, workspaceSize, executor);
    }
    if (paddedMode) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "paddedMode only supports false currently, but got true.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclnnStatus ret = MoeTokenUnpermuteCheck::CheckParams(permutedTokens, sortedIndices, probsOptional, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入转换成连续的tensor
    auto permutedTokensContiguous = l0op::Contiguous(permutedTokens, uniqueExecutor.get());
    CHECK_RET(permutedTokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sortedIndicesContiguous = l0op::Contiguous(sortedIndices, uniqueExecutor.get());
    CHECK_RET(sortedIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *probsContiguous = nullptr;
    if (probsOptional != nullptr) {
        probsContiguous = l0op::Contiguous(probsOptional, uniqueExecutor.get());
        CHECK_RET(probsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto out_ = l0op::MoeFinalizeRoutingV2(permutedTokensContiguous, sortedIndicesContiguous, nullptr, nullptr, nullptr,
                                           probsContiguous, nullptr, nullptr, nullptr, nullptr, nullptr,
                                           READ_INDEX_BY_ROW, nullptr, nullptr, nullptr, 1, out, uniqueExecutor.get());
    CHECK_RET(out_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // copyout结果，如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyOutResult = l0op::ViewCopy(out_, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenUnpermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    static bool useMoeFinalizeRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2) {
        return aclnnInnerMoeTokenUnpermute(workspace, workspaceSize, executor, stream);
    }

    L2_DFX_PHASE_2(aclnnMoeTokenUnpermute);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
