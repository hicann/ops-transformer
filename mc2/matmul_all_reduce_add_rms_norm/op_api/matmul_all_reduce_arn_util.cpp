/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_all_reduce_arn_util.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "aclnnInner_matmul_all_reduce_add_rms_norm.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void* executor, const size_t irIndex);

// 根据API定义，需要列出所能支持的所有dtype
const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_310P = {op::DataType::DT_FLOAT16};

// quant
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BIAS = {op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT = {op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_PERTOKEN = {op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT_310P = {op::DataType::DT_UINT64,
                                                                                    op::DataType::DT_INT64};

bool ArnCheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* gamma)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(residual, return false);
    OP_CHECK_NULL(gamma, return false);
    return true;
}

bool ArnCheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual)
{
    const size_t x1_len = x1->GetViewShape().GetDimNum();
    const size_t x2_len = x2->GetViewShape().GetDimNum();
    const size_t residual_len = residual->GetViewShape().GetDimNum();
    OP_LOGI("MatmulAllReduceAddRmsNorm, x1 shape: %s, x2 shape: %s, residual shape: %s.",
            op::ToString(x1->GetViewShape()).GetString(), op::ToString(x2->GetViewShape()).GetString(),
            op::ToString(residual->GetViewShape()).GetString());
    if (x1_len < NUM_ONE || x2_len < NUM_ONE || residual_len < NUM_ONE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize", "x1/x2/residual", "1D or less",
                                     "at least 2D");
        return false;
    }
    if (x1->GetViewShape().GetDim(x1_len - 1) != x2->GetViewShape().GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize", "x1/x2",
                                  op::ToString(x1->GetViewShape()).GetString(),
                                  op::ToString(x2->GetViewShape()).GetString());
        return false;
    }
    if (residual->GetViewShape().GetDim(residual_len - 1) != x2->GetViewShape().GetDim(x2_len - 1)) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize", "residual/x2",
                                  op::ToString(residual->GetViewShape()).GetString(),
                                  op::ToString(x2->GetViewShape()).GetString());
        return false;
    }
    return true;
}

// AddRmsNorm Inner
aclnnStatus InnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffset, const aclTensor* dequant, const aclTensor* residual, const aclTensor* gamma,
    double epsilon, const char* group, const char* reduceOp, int64_t commTurn, int64_t antiquantGroupSize,
    const aclTensor* y, const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2);
    aclnnStatus ret = aclnnInnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
        x1, x2, bias, residual, gamma, antiquantScale, antiquantOffset, dequant, const_cast<char*>(group),
        const_cast<char*>(reduceOp), transposeX1, transposeX2, commTurn, antiquantGroupSize, epsilon, y, normOut,
        workspaceSize, executor);
    OP_LOGI("Group %s, reduce op %s, trans flag %u %u, epsilon %lf, ret %d.", group, reduceOp, transposeX1, transposeX2,
            epsilon, ret);
#ifdef MC2_UT
    ret = 0;
#endif
    if (ret == OK) {
        if (NnopbaseDisableOptionalInput != nullptr) {
            if (antiquantScale == nullptr) {
                NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
                NnopbaseDisableOptionalInput(*executor, 6U); // 6 is input irIndex
            }
            NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
        }
    }
    return ret;
}

#ifdef __cplusplus
}
#endif
