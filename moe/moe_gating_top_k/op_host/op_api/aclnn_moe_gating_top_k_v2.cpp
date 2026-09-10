/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_gating_top_k_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe_gating_top_k.h"

using namespace op;

namespace MoeGatingTopKV2Check {

static const std::initializer_list<op::DataType> MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_INT = {DataType::DT_INT32,
                                                                                               DataType::DT_INT64};
static const std::initializer_list<op::DataType> MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_INT32 = {DataType::DT_INT32};
static const std::initializer_list<op::DataType> MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_FLOAT = {DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensor *x, const aclTensor *yOut, const aclTensor *expertIdxOut,
                                const aclTensor *outOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(expertIdxOut, return false);
    OP_CHECK_NULL(outOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *x, const aclTensor *biasOptional, const aclTensor *inputIdsOptional,
                                   const aclTensor *tid2eidOptional, const aclTensor *yOut,
                                   const aclTensor *expertIdxOut, const aclTensor *outOut)
{
    if (x != nullptr && x->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x, MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (biasOptional != nullptr && biasOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(x, biasOptional, return false);
    }
    if (inputIdsOptional != nullptr && inputIdsOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputIdsOptional, MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_INT, return false);
    }
    if (tid2eidOptional != nullptr && tid2eidOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(tid2eidOptional, MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_INT, return false);
    }
    if (yOut != nullptr && yOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(yOut, MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(x, yOut, return false);
    }
    if (expertIdxOut != nullptr && expertIdxOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdxOut, MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_INT32, return false);
    }
    if (outOut != nullptr && outOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(outOut, MOE_GATING_TOP_K_V2_DTYPE_SUPPORT_LIST_FLOAT, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *biasOptional, const aclTensor *inputIdsOptional,
                               const aclTensor *tid2eidOptional, const aclTensor *yOut, const aclTensor *expertIdxOut,
                               const aclTensor *outOut)
{
    CHECK_RET(CheckNotNull(x, yOut, expertIdxOut, outOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, biasOptional, inputIdsOptional, tid2eidOptional, yOut, expertIdxOut, outOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeGatingTopKV2Check

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMoeGatingTopKV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *biasOptional, const aclTensor *inputIdsOptional,
    const aclTensor *tid2eidOptional, int64_t k, int64_t kGroup, int64_t groupCount, int64_t groupSelectMode,
    int64_t renorm, int64_t normType, bool outFlag, double routedScalingFactor, double eps, const aclTensor *yOut,
    const aclTensor *expertIdxOut, const aclTensor *outOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMoeGatingTopKV2,
                   DFX_IN(x, biasOptional, inputIdsOptional, tid2eidOptional, k, kGroup, groupCount, groupSelectMode,
                          renorm, normType, outFlag, routedScalingFactor, eps),
                   DFX_OUT(yOut, expertIdxOut, outOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclnnStatus ret = MoeGatingTopKV2Check::CheckParams(x, biasOptional, inputIdsOptional, tid2eidOptional, yOut,
                                                        expertIdxOut, outOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *biasContiguous = nullptr;
    if (biasOptional != nullptr) {
        biasContiguous = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *inputIdsContiguous = nullptr;
    if (inputIdsOptional != nullptr) {
        inputIdsContiguous = l0op::Contiguous(inputIdsOptional, uniqueExecutor.get());
        CHECK_RET(inputIdsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *tid2eidContiguous = nullptr;
    if (tid2eidOptional != nullptr) {
        tid2eidContiguous = l0op::Contiguous(tid2eidOptional, uniqueExecutor.get());
        CHECK_RET(tid2eidContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto result = l0op::MoeGatingTopK(xContiguous, biasContiguous, inputIdsContiguous, tid2eidContiguous, k, kGroup,
                                      groupCount, groupSelectMode, renorm, normType, outFlag, routedScalingFactor, eps,
                                      yOut, expertIdxOut, outOut, uniqueExecutor.get());
    CHECK_RET(result.y != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(result.expertIdx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(result.outOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyYOutResult = l0op::ViewCopy(result.y, yOut, uniqueExecutor.get());
    CHECK_RET(viewCopyYOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyExpertIdxResult = l0op::ViewCopy(result.expertIdx, expertIdxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpertIdxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyOutOutResult = l0op::ViewCopy(result.outOut, outOut, uniqueExecutor.get());
    CHECK_RET(viewCopyOutOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeGatingTopKV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeGatingTopKV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
