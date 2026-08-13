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
#include "moe_finalize_routing_common.h"

using namespace op;

static aclnnStatus CheckV2(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                           const aclTensor *x1Optional, const aclTensor *x2Optional,
                           const aclTensor *biasOptional, const aclTensor *scalesOptional,
                           const aclTensor *expertIdxOptional, const aclTensor *out, int64_t dropPadMode)
{
    if (MoeFinalizeRoutingCheck::Is310P()) {
        return MoeFinalizeRoutingCheck::CheckParams310P(expandedX, expandedRowIdx, x1Optional, x2Optional,
                                                        biasOptional, scalesOptional, expertIdxOptional, out,
                                                        dropPadMode);
    } else if (MoeFinalizeRoutingCheck::IsCommonValidationChip()) {
        return MoeFinalizeRoutingCheck::CheckParams(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional,
                                                    scalesOptional, expertIdxOptional, nullptr, nullptr, nullptr,
                                                    nullptr, out, dropPadMode);
    }
    return ACLNN_SUCCESS;
}

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

    auto ret = CheckV2(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                       expertIdxOptional, out, dropPadMode);
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
