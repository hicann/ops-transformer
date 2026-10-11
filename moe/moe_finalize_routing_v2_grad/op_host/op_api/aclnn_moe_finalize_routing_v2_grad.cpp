/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_finalize_routing_v2_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe_finalize_routing_v2_grad.h"

using namespace op;

namespace MoeFinalizeRoutingV2GradCheck {

static const std::initializer_list<op::DataType> MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX = {
    DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* gradY, const aclTensor* expandedRowIdx,
                                const aclTensor* gradExpandedXOut, const aclTensor* gradScalesOut)
{
    OP_CHECK_NULL(gradY, return false);
    OP_CHECK_NULL(expandedRowIdx, return false);
    OP_CHECK_NULL(gradExpandedXOut, return false);
    OP_CHECK_NULL(gradScalesOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* gradY, const aclTensor* expandedRowIdx,
                                   const aclTensor* expandedXOptional, const aclTensor* scalesOptional,
                                   const aclTensor* expertIdxOptional, const aclTensor* biasOptional,
                                   const aclTensor* gradExpandedXOut, const aclTensor* gradScalesOut)
{
    if (gradY != nullptr && gradY->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradY, MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expandedRowIdx != nullptr && expandedRowIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdx, MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX,
                                   return false);
    }
    if (expandedXOptional != nullptr && expandedXOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedXOptional, MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(gradY, expandedXOptional, return false);
    }
    if (scalesOptional != nullptr && scalesOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalesOptional, MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expertIdxOptional != nullptr && expertIdxOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdxOptional, MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX,
                                   return false);
    }
    if (biasOptional != nullptr && biasOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(gradY, biasOptional, return false);
    }
    if (gradExpandedXOut != nullptr && gradExpandedXOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradExpandedXOut, MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(gradY, gradExpandedXOut, return false);
    }
    if (gradScalesOut != nullptr && gradScalesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradScalesOut, MOE_FINALIZE_ROUTING_V2_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradY, const aclTensor* expandedRowIdx,
                               const aclTensor* expandedXOptional, const aclTensor* scalesOptional,
                               const aclTensor* expertIdxOptional, const aclTensor* biasOptional,
                               const aclTensor* gradExpandedXOut, const aclTensor* gradScalesOut)
{
    CHECK_RET(CheckNotNull(gradY, expandedRowIdx, gradExpandedXOut, gradScalesOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional,
                              gradExpandedXOut, gradScalesOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeFinalizeRoutingV2GradCheck

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* expandedRowIdx, const aclTensor* expandedXOptional,
    const aclTensor* scalesOptional, const aclTensor* expertIdxOptional, const aclTensor* biasOptional,
    int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity,
    const aclTensor* gradExpandedXOut, const aclTensor* gradScalesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeFinalizeRoutingV2Grad,
                   DFX_IN(gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional,
                          dropPadMode, activeNum, expertNum, expertCapacity),
                   DFX_OUT(gradExpandedXOut, gradScalesOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclnnStatus ret =
        MoeFinalizeRoutingV2GradCheck::CheckParams(gradY, expandedRowIdx, expandedXOptional, scalesOptional,
                                                   expertIdxOptional, biasOptional, gradExpandedXOut, gradScalesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，将输入转换成连续的tensor
    auto gradYContiguous = l0op::Contiguous(gradY, uniqueExecutor.get());
    CHECK_RET(gradYContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto expandedRowIdxContiguous = l0op::Contiguous(expandedRowIdx, uniqueExecutor.get());
    CHECK_RET(expandedRowIdxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* expandedXContiguous = nullptr;
    if (expandedXOptional != nullptr) {
        expandedXContiguous = l0op::Contiguous(expandedXOptional, uniqueExecutor.get());
        CHECK_RET(expandedXContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* scalesContiguous = nullptr;
    if (scalesOptional != nullptr) {
        scalesContiguous = l0op::Contiguous(scalesOptional, uniqueExecutor.get());
        CHECK_RET(scalesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* expertIdxContiguous = nullptr;
    if (expertIdxOptional != nullptr) {
        expertIdxContiguous = l0op::Contiguous(expertIdxOptional, uniqueExecutor.get());
        CHECK_RET(expertIdxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* biasContiguous = nullptr;
    if (biasOptional != nullptr) {
        biasContiguous = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0接口进行计算，传入输出参数
    auto result =
        l0op::MoeFinalizeRoutingV2Grad(gradYContiguous, expandedRowIdxContiguous, expandedXContiguous, scalesContiguous,
                                       expertIdxContiguous, biasContiguous, dropPadMode, activeNum, expertNum,
                                       expertCapacity, gradExpandedXOut, gradScalesOut, uniqueExecutor.get());
    auto [gradExpandedXOut_, gradScalesOut_] = result;
    bool hasNullptr = (gradExpandedXOut_ == nullptr) || (gradScalesOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyGradExpandedXOutResult = l0op::ViewCopy(gradExpandedXOut_, gradExpandedXOut, uniqueExecutor.get());
    CHECK_RET(viewCopyGradExpandedXOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyGradScalesOutResult = l0op::ViewCopy(gradScalesOut_, gradScalesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyGradScalesOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV2Grad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeFinalizeRoutingV2Grad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
