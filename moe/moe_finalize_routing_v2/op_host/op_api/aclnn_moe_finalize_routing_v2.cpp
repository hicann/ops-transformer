/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_finalize_routing_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe_finalize_routing_v2.h"

using namespace op;

namespace MoeFinalizeRoutingV2Check {

static const std::initializer_list<op::DataType> MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX = {
    DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *expandedX, const aclTensor *expandedRowIdx, const aclTensor *out)
{
    OP_CHECK_NULL(expandedX, return false);
    OP_CHECK_NULL(expandedRowIdx, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                   const aclTensor *x1Optional, const aclTensor *x2Optional,
                                   const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                   const aclTensor *expertIdxOptional, const aclTensor *out)
{
    if (expandedX != nullptr && expandedX->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedX, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expandedRowIdx != nullptr && expandedRowIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdx, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (x1Optional != nullptr && x1Optional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1Optional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, x1Optional, return false);
    }
    if (x2Optional != nullptr && x2Optional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x2Optional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, x2Optional, return false);
    }
    if (biasOptional != nullptr && biasOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, biasOptional, return false);
    }
    if (scalesOptional != nullptr && scalesOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalesOptional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expertIdxOptional != nullptr && expertIdxOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdxOptional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (out != nullptr && out->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(out, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, out, return false);
    }
    return true;
}

static inline bool CheckDtypeValid310P(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                       const aclTensor *x1Optional, const aclTensor *x2Optional,
                                       const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                       const aclTensor *expertIdxOptional, const aclTensor *out)
{
    if (expandedX != nullptr && expandedX->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedX, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P, return false);
    }
    if (expandedRowIdx != nullptr && expandedRowIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdx, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (x1Optional != nullptr && x1Optional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1Optional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, x1Optional, return false);
    }
    if (x2Optional != nullptr && x2Optional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x2Optional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, x2Optional, return false);
    }
    if (biasOptional != nullptr && biasOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, biasOptional, return false);
    }
    if (scalesOptional != nullptr && scalesOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalesOptional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P, return false);
    }
    if (expertIdxOptional != nullptr && expertIdxOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdxOptional, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (out != nullptr && out->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(out, MOE_FINALIZE_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, out, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *expandedX, const aclTensor *expandedRowIdx, const aclTensor *x1Optional,
                               const aclTensor *x2Optional, const aclTensor *biasOptional,
                               const aclTensor *scalesOptional, const aclTensor *expertIdxOptional,
                               const aclTensor *out)
{
    CHECK_RET(CheckNotNull(expandedX, expandedRowIdx, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                              expertIdxOptional, out),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams310P(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                   const aclTensor *x1Optional, const aclTensor *x2Optional,
                                   const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                   const aclTensor *expertIdxOptional, const aclTensor *out)
{
    CHECK_RET(CheckNotNull(expandedX, expandedRowIdx, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid310P(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                                  expertIdxOptional, out),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeFinalizeRoutingV2Check

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(
    const aclTensor *expandedX, const aclTensor *expandedRowIdx, const aclTensor *x1Optional,
    const aclTensor *x2Optional, const aclTensor *biasOptional, const aclTensor *scalesOptional,
    const aclTensor *expertIdxOptional, int64_t dropPadMode, const aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeFinalizeRoutingV2,
                   DFX_IN(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                          expertIdxOptional, dropPadMode),
                   DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
    bool isCommonValidationChip = Ops::Transformer::AclnnUtil::IsRegbase() ||
                                  GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                                  GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93;

    aclnnStatus ret = ACLNN_SUCCESS;
    if (is310P) {
        ret = MoeFinalizeRoutingV2Check::CheckParams310P(expandedX, expandedRowIdx, x1Optional, x2Optional,
                                                         biasOptional, scalesOptional, expertIdxOptional, out);
    } else if (isCommonValidationChip) {
        ret = MoeFinalizeRoutingV2Check::CheckParams(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional,
                                                     scalesOptional, expertIdxOptional, out);
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，将输入转换成连续的tensor
    auto expandedXContiguous = l0op::Contiguous(expandedX, uniqueExecutor.get());
    CHECK_RET(expandedXContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto expandedRowIdxContiguous = l0op::Contiguous(expandedRowIdx, uniqueExecutor.get());
    CHECK_RET(expandedRowIdxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *x1Contiguous = nullptr;
    if (x1Optional != nullptr) {
        x1Contiguous = l0op::Contiguous(x1Optional, uniqueExecutor.get());
        CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *x2Contiguous = nullptr;
    if (x2Optional != nullptr) {
        x2Contiguous = l0op::Contiguous(x2Optional, uniqueExecutor.get());
        CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *biasContiguous = nullptr;
    if (biasOptional != nullptr) {
        biasContiguous = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *scalesContiguous = nullptr;
    if (scalesOptional != nullptr) {
        scalesContiguous = l0op::Contiguous(scalesOptional, uniqueExecutor.get());
        CHECK_RET(scalesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *expertIdxContiguous = nullptr;
    if (expertIdxOptional != nullptr) {
        expertIdxContiguous = l0op::Contiguous(expertIdxOptional, uniqueExecutor.get());
        CHECK_RET(expertIdxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0接口进行计算，传入out参数
    auto out_ =
        l0op::MoeFinalizeRoutingV2(expandedXContiguous, expandedRowIdxContiguous, x1Contiguous, x2Contiguous,
                                   biasContiguous, scalesContiguous, expertIdxContiguous, nullptr, nullptr, nullptr,
                                   nullptr, dropPadMode, nullptr, nullptr, nullptr, 1, out, uniqueExecutor.get());
    CHECK_RET(out_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyOutResult = l0op::ViewCopy(out_, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeFinalizeRoutingV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
