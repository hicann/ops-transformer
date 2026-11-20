/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_floyd_attention_grad.h"
#include <utility>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

namespace {
extern "C" aclnnStatus aclnnInnerFusedFloydAttentionGradGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key1, const aclTensor* value1, const aclTensor* key2,
    const aclTensor* value2, const aclTensor* dy, const aclTensor* attenMaskOptional,
    const aclTensor* softmaxMaxOptional, const aclTensor* softmaxSumOptional, const aclTensor* attentionInOptional,
    double scaleValue, const aclTensor* dqOut, const aclTensor* dk1Out, const aclTensor* dv1Out,
    const aclTensor* dk2Out, const aclTensor* dv2Out, uint64_t* workspaceSize, aclOpExecutor** executor);
extern "C" aclnnStatus aclnnInnerFusedFloydAttentionGrad(void* workspace, uint64_t workspaceSize,
                                                         aclOpExecutor* executor, aclrtStream stream);

aclnnStatus CheckTensorDataType(const aclTensor* tensor, op::DataType expectedType, const char* name)
{
    // Preserve the generated inner interface's handling of absent optional inputs.
    if (tensor != nullptr && tensor->GetDataType() != expectedType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dtype is invalid: got %s, expected %s.", name,
                op::ToString(tensor->GetDataType()).GetString(), op::ToString(expectedType).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputContiguous(const aclTensor* tensor, const char* name)
{
    // The kernel writes outputs to GM by DataCopy with contiguous layout and ignores the view strides,
    // so a non-contiguous output would silently keep stale data (see issue #7476). Reject it early.
    if (tensor != nullptr && !IsContiguous(tensor)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be contiguous: non-contiguous outputs are not supported.", name);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnFusedFloydAttentionGradGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key1, const aclTensor* value1, const aclTensor* key2,
    const aclTensor* value2, const aclTensor* dy, const aclTensor* attenMaskOptional,
    const aclTensor* softmaxMaxOptional, const aclTensor* softmaxSumOptional, const aclTensor* attentionInOptional,
    double scaleValue, const aclTensor* dqOut, const aclTensor* dk1Out, const aclTensor* dv1Out,
    const aclTensor* dk2Out, const aclTensor* dv2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckTensorDataType(softmaxMaxOptional, op::DataType::DT_FLOAT, "softmaxMax");
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = CheckTensorDataType(softmaxSumOptional, op::DataType::DT_FLOAT, "softmaxSum");
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (query != nullptr) {
        ret = CheckTensorDataType(attentionInOptional, query->GetDataType(), "attentionIn (must match query)");
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    for (const auto& out :
         {std::make_pair(dqOut, "dqOut"), std::make_pair(dk1Out, "dk1Out"), std::make_pair(dv1Out, "dv1Out"),
          std::make_pair(dk2Out, "dk2Out"), std::make_pair(dv2Out, "dv2Out")}) {
        ret = CheckOutputContiguous(out.first, out.second);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    return aclnnInnerFusedFloydAttentionGradGetWorkspaceSize(
        query, key1, value1, key2, value2, dy, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional,
        attentionInOptional, scaleValue, dqOut, dk1Out, dv1Out, dk2Out, dv2Out, workspaceSize, executor);
}

aclnnStatus aclnnFusedFloydAttentionGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    return aclnnInnerFusedFloydAttentionGrad(workspace, workspaceSize, executor, stream);
}
