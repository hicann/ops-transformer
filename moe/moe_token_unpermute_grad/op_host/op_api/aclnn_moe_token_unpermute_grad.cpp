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
 * \file aclnn_moe_token_unpermute_grad.cpp
 * \brief
 */
#include "aclnn_moe_token_unpermute_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_token_unpermute_grad.h" // 该文件为自动生成，在build/autogen/inner路径下
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe/moe_finalize_routing_v2_grad/op_host/op_api/moe_finalize_routing_v2_grad.h"

using namespace op;

namespace MoeTokenUnpermuteGradCheck {

static const std::initializer_list<op::DataType> MOE_GRAD_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                                const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut,
                                const aclTensor *permuteTokens)
{
    OP_CHECK_NULL(permuteTokens, return false);
    OP_CHECK_NULL(unpermutedTokensGrad, return false);
    OP_CHECK_NULL(sortedIndices, return false);
    OP_CHECK_NULL(permutedTokensGradOut, return false);
    OP_CHECK_NULL(probsGradOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                                   const aclTensor *permuteTokens, const aclTensor *probsOptional,
                                   const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut)
{
    if (unpermutedTokensGrad != nullptr && unpermutedTokensGrad->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(unpermutedTokensGrad, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (sortedIndices != nullptr && sortedIndices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndices, MOE_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (permuteTokens != nullptr && permuteTokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokens, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(unpermutedTokensGrad, permuteTokens, return false);
    }
    if (probsOptional != nullptr && probsOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsOptional, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (permutedTokensGradOut != nullptr && permutedTokensGradOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permutedTokensGradOut, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(unpermutedTokensGrad, permutedTokensGradOut, return false);
    }
    if (probsGradOut != nullptr && probsGradOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsGradOut, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                               const aclTensor *permuteTokens, const aclTensor *probsOptional,
                               const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut)
{
    CHECK_RET(CheckNotNull(unpermutedTokensGrad, sortedIndices, permutedTokensGradOut, probsGradOut, permuteTokens),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(unpermutedTokensGrad, sortedIndices, permuteTokens, probsOptional, permutedTokensGradOut,
                              probsGradOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenUnpermuteGradCheck

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(const aclTensor *permuteTokensOptional,
                                                       const aclTensor *unpermutedTokensGrad,
                                                       const aclTensor *sortedIndices, const aclTensor *probsOptional,
                                                       bool paddedMode, const aclIntArray *restoreShapeOptional,
                                                       aclTensor *permutedTokensGradOut, aclTensor *probsGradOut,
                                                       uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenUnpermuteGrad,
                   DFX_IN(permuteTokensOptional, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode,
                          restoreShapeOptional),
                   DFX_OUT(permutedTokensGradOut, probsGradOut));

    bool useMoeFinalizeRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2Grad) {
        return aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
            permuteTokensOptional, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode, restoreShapeOptional,
            permutedTokensGradOut, probsGradOut, workspaceSize, executor);
    }
    if (paddedMode) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "paddedMode only supports false currently, but got true.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclnnStatus ret = MoeTokenUnpermuteGradCheck::CheckParams(
        unpermutedTokensGrad, sortedIndices, permuteTokensOptional, probsOptional, permutedTokensGradOut, probsGradOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入转换成连续的tensor
    auto unpermutedTokensGradContiguous = l0op::Contiguous(unpermutedTokensGrad, uniqueExecutor.get());
    CHECK_RET(unpermutedTokensGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sortedIndicesContiguous = l0op::Contiguous(sortedIndices, uniqueExecutor.get());
    CHECK_RET(sortedIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *permuteTokensContiguous = l0op::Contiguous(permuteTokensOptional, uniqueExecutor.get());
    CHECK_RET(permuteTokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *probsContiguous = nullptr;
    if (probsOptional != nullptr) {
        probsContiguous = l0op::Contiguous(probsOptional, uniqueExecutor.get());
        CHECK_RET(probsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 获取activeNum
    auto sortedIndicesShape = sortedIndicesContiguous->GetViewShape();
    CHECK_RET(sortedIndicesShape.GetDimNum() > 0, ACLNN_ERR_PARAM_INVALID);
    int64_t activeNum = sortedIndicesShape.GetDim(0);

    // 调用l0接口进行计算
    auto result = l0op::MoeFinalizeRoutingV2Grad(
        unpermutedTokensGradContiguous, sortedIndicesContiguous, permuteTokensContiguous, probsContiguous, nullptr,
        nullptr, 0, activeNum, 0, 0, permutedTokensGradOut, probsGradOut, uniqueExecutor.get());
    auto [gradExpandedXOut_, gradScalesOut_] = result;
    bool hasNullptr = (gradExpandedXOut_ == nullptr) || (gradScalesOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyGradExpandedXOutResult =
        l0op::ViewCopy(gradExpandedXOut_, permutedTokensGradOut, uniqueExecutor.get());
    CHECK_RET(viewCopyGradExpandedXOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyGradScalesOutResult = l0op::ViewCopy(gradScalesOut_, probsGradOut, uniqueExecutor.get());
    CHECK_RET(viewCopyGradScalesOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenUnpermuteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    static bool useMoeFinalizeRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2Grad) {
        return aclnnInnerMoeTokenUnpermuteGrad(workspace, workspaceSize, executor, stream);
    }

    L2_DFX_PHASE_2(aclnnMoeTokenUnpermuteGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
