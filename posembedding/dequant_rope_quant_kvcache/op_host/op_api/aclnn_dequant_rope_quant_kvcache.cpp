/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_dequant_rope_quant_kvcache.h"
#include <string>
#include "aclnnInner_dequant_rope_quant_kvcache.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "log/log.h"

namespace {
constexpr const char* API_NAME = "aclnnDequantRopeQuantKvcacheGetWorkspaceSize";

aclnnStatus CheckRopeDtypes(const aclTensor* x, const aclTensor* cos, const aclTensor* sin)
{
    const auto xType = x->GetDataType();
    const auto cosType = cos->GetDataType();
    const auto sinType = sin->GetDataType();
    if (xType != op::DataType::DT_FLOAT16 && xType != op::DataType::DT_BF16 && xType != op::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(API_NAME, "x", op::ToString(xType).GetString(), "FLOAT16, BFLOAT16 or INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // INT32 dequantization supports either floating-point RoPE dtype, independently of x.
    const auto expectedType = xType == op::DataType::DT_INT32 ? cosType : xType;
    if ((expectedType == op::DataType::DT_FLOAT16 || expectedType == op::DataType::DT_BF16) &&
        cosType == expectedType && sinType == expectedType) {
        return ACLNN_SUCCESS;
    }
    const std::string actual = std::string("Actual dtypes: x.dtype=") + op::ToString(xType).GetString() +
                               ", cos.dtype=" + op::ToString(cosType).GetString() +
                               ", sin.dtype=" + op::ToString(sinType).GetString();
    const std::string constraint =
        xType == op::DataType::DT_INT32 ?
            "cos.dtype and sin.dtype must match and be FLOAT16 or BFLOAT16 when x.dtype is INT32" :
            "x.dtype, cos.dtype and sin.dtype must be consistent when x.dtype is FLOAT16 or BFLOAT16";
    const std::string reason = constraint + ". " + actual;
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "x, cos, sin", reason.c_str());
    return ACLNN_ERR_PARAM_INVALID;
}
} // namespace

aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(
    const aclTensor* x, const aclTensor* cos, const aclTensor* sin, aclTensor* kCacheRef, aclTensor* vCacheRef,
    const aclTensor* indices, const aclTensor* scaleK, const aclTensor* scaleV, const aclTensor* offsetKOptional,
    const aclTensor* offsetVOptional, const aclTensor* weightScaleOptional, const aclTensor* activationScaleOptional,
    const aclTensor* biasOptional, const aclIntArray* sizeSplits, char* quantModeOptional, char* layoutOptional,
    bool kvOutput, char* cacheModeOptional, const aclTensor* qOut, const aclTensor* kOut, const aclTensor* vOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // These are the only tensors inspected before delegating to the generated interface.
    const char* nullParam = nullptr;
    if (x == nullptr) {
        nullParam = "x";
    } else if (cos == nullptr) {
        nullParam = "cos";
    } else if (sin == nullptr) {
        nullParam = "sin";
    } else if (workspaceSize == nullptr) {
        nullParam = "workspaceSize";
    } else if (executor == nullptr) {
        nullParam = "executor";
    }
    if (nullParam != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, nullParam, "nullptr", "must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;
    const aclnnStatus ret = CheckRopeDtypes(x, cos, sin);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    return aclnnInnerDequantRopeQuantKvcacheGetWorkspaceSize(
        x, cos, sin, kCacheRef, vCacheRef, indices, scaleK, scaleV, offsetKOptional, offsetVOptional,
        weightScaleOptional, activationScaleOptional, biasOptional, sizeSplits, quantModeOptional, layoutOptional,
        kvOutput, cacheModeOptional, qOut, kOut, vOut, workspaceSize, executor);
}

aclnnStatus aclnnDequantRopeQuantKvcache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    return aclnnInnerDequantRopeQuantKvcache(workspace, workspaceSize, executor, stream);
}
