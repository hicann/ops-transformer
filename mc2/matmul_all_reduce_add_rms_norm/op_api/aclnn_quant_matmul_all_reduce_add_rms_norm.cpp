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
 * \file aclnn_quant_matmul_all_reduce_add_rms_norm.cpp
 * \brief
 */

#include "aclnn_quant_matmul_all_reduce_add_rms_norm.h"
#include "securec.h"

#include "acl/acl.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mc2_log_compat.h"
#include "aclnn_kernels/contiguous.h"
#include "matmul_all_reduce_arn_util.h"
#include "aclnnInner_matmul_all_reduce_add_rms_norm.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                                const aclTensor* bias, const aclTensor* dequantScale,
                                                                const aclTensor* residual, const aclTensor* gamma,
                                                                double epsilon, const char* group, const char* reduceOp,
                                                                int64_t commTurn, int64_t streamMode,
                                                                const aclTensor* y, const aclTensor* normOut,
                                                                uint64_t* workspaceSize, aclOpExecutor** executor)
{
    DEPRECATED_MM_ARN_API_WARN_ONCE("aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize",
                                    "aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize", "December 2026",
                                    "aclnnQuantMatmulAllReduceV2GetWorkspaceSize", "aclnnAddRmsNormGetWorkspaceSize");
    CHECK_RET(ArnCheckNotNull(x1, x2, residual, gamma), ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(dequantScale, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(ArnCheckShape(x1, x2, residual), ACLNN_ERR_PARAM_INVALID);
    // dequantScale转为uint64
    auto dequant = const_cast<aclTensor*>(dequantScale);
    if (dequant == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize", "dequantScale");
        return ACLNN_ERR_INNER;
    }
    if (dequant->GetDataType() == op::DataType::DT_INT64) {
        dequant->SetDataType(op::DataType::DT_UINT64);
    }
    // 处理空tensor,x1,x2不为空，dequantscale为空也报错，bias、x3可选不做判断
    if (x1->IsEmpty() || x2->IsEmpty() || dequant->IsEmpty() || residual->IsEmpty() || gamma->IsEmpty()) {
        // 根据实际支持情况补充
        OP_LOGD("QuantMatmulAllReduceAddRmsNorm, dealing with empty tensor.");
        // 处理k = 0 场景，报错
        int64_t kValue = x1->GetViewShape().GetDim(x1->GetViewShape().GetDimNum() - 1);
        OP_LOGD("QuantMatmulAllReduceAddRmsNorm, kValue: %ld.", kValue);
        if (kValue == 0) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize", "x1 k-dim", "0",
                                      "non-zero");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 固定写法，创建OpExecutor
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // check streamMode = NUM_ACL_STOP_ON_FAILURE
    CHECK_RET(streamMode == NUM_ACL_STOP_ON_FAILURE, ACLNN_ERR_PARAM_INVALID);
    int64_t antiquantGroupSize = 0;
    aclnnStatus ret = InnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
        x1, x2, bias, nullptr, nullptr, dequant, residual, gamma, epsilon, group, reduceOp, commTurn,
        antiquantGroupSize, y, normOut, workspaceSize, executor);
    OP_LOGD("QuantMatmulAllReduceAddRmsNorm, end ret %d", ret);
    return ret;
}

aclnnStatus aclnnQuantMatmulAllReduceAddRmsNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                const aclrtStream stream)
{
    DEPRECATED_MM_ARN_API_WARN_ONCE("aclnnQuantMatmulAllReduceAddRmsNorm", "aclnnInplaceQuantMatmulAllReduceAddRmsNorm",
                                    "December 2026", "aclnnQuantMatmulAllReduceV2", "aclnnAddRmsNorm");
    if (workspace == nullptr || workspaceSize == 0UL) {
        OP_LOGD("Skip the api for empty tensor, workspace addr %p, size %lu.", workspace, workspaceSize);
        return ACLNN_SUCCESS;
    }

    aclnnStatus ret = aclnnInnerMatmulAllReduceAddRmsNorm(workspace, workspaceSize, executor, stream);
    OP_LOGD("QuantMatmulAllReduceAddRmsNorm, aclnnQuantMatmulAllReduceAddRmsNorm ret %d", ret);
    if (ret != 0) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnQuantMatmulAllReduceAddRmsNorm", "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
