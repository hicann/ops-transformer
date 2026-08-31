/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_token_permute_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_token_permute_grad.h" // 该文件为自动生成，在build/autogen/inner路径下
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe/moe_init_routing_v2_grad/op_host/op_api/moe_init_routing_v2_grad.h"

using namespace op;

namespace MoeTokenPermuteGradCheck {

static const std::initializer_list<op::DataType> MOE_GRAD_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *permutedOutputGrad, const aclTensor *sortedIndices,
                                const aclTensor *out)
{
    OP_CHECK_NULL(permutedOutputGrad, return false);
    OP_CHECK_NULL(sortedIndices, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *permutedOutputGrad, const aclTensor *sortedIndices,
                                   const aclTensor *out)
{
    if (permutedOutputGrad != nullptr && permutedOutputGrad->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permutedOutputGrad, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (sortedIndices != nullptr && sortedIndices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndices, MOE_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (out != nullptr && out->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(out, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(permutedOutputGrad, out, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *permutedOutputGrad, const aclTensor *sortedIndices,
                               const aclTensor *out)
{
    CHECK_RET(CheckNotNull(permutedOutputGrad, sortedIndices, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(permutedOutputGrad, sortedIndices, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenPermuteGradCheck

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(const aclTensor *permutedOutputGrad,
                                                     const aclTensor *sortedIndices, int64_t numTopk, bool paddedMode,
                                                     aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenPermuteGrad, DFX_IN(permutedOutputGrad, sortedIndices, numTopk, paddedMode),
                   DFX_OUT(out));

    bool useMoeInitRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2Grad) {
        return aclnnInnerMoeTokenPermuteGradGetWorkspaceSize(permutedOutputGrad, sortedIndices, numTopk, paddedMode,
                                                             out, workspaceSize, executor);
    }
    if (paddedMode) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "paddedMode only supports false currently, but got true.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclnnStatus ret = MoeTokenPermuteGradCheck::CheckParams(permutedOutputGrad, sortedIndices, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入转换成连续的tensor
    auto permutedOutputGradContiguous = l0op::Contiguous(permutedOutputGrad, uniqueExecutor.get());
    CHECK_RET(permutedOutputGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sortedIndicesContiguous = l0op::Contiguous(sortedIndices, uniqueExecutor.get());
    CHECK_RET(sortedIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取activeNum
    auto permutedOutputGradShape = permutedOutputGrad->GetViewShape();
    CHECK_RET(permutedOutputGradShape.GetDimNum() > 0, ACLNN_ERR_PARAM_INVALID);
    int64_t activeNum = permutedOutputGradShape.GetDim(0);

    // 调用l0接口进行计算
    auto out_ = l0op::MoeInitRoutingV2Grad(permutedOutputGradContiguous, sortedIndicesContiguous, numTopk, 0, activeNum,
                                           out, uniqueExecutor.get());
    CHECK_RET(out_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyOutResult = l0op::ViewCopy(out_, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenPermuteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    static bool useMoeInitRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2Grad) {
        return aclnnInnerMoeTokenPermuteGrad(workspace, workspaceSize, executor, stream);
    }

    L2_DFX_PHASE_2(aclnnMoeTokenPermuteGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
