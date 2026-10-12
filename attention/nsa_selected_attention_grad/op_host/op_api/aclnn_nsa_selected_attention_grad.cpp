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
 * \file aclnn_nsa_selected_attention_grad.cpp
 * \brief
 */

#include "aclnn_nsa_selected_attention_grad.h"
#include "nsa_selected_attention_grad.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct NsaSelectedAttentionGradParams {
    const aclTensor* query = nullptr;
    const aclTensor* key = nullptr;
    const aclTensor* value = nullptr;
    const aclTensor* attentionOut = nullptr;
    const aclTensor* attentionOutGrad = nullptr;
    const aclTensor* softmaxMax = nullptr;
    const aclTensor* softmaxSum = nullptr;
    const aclTensor* topkIndices = nullptr;
    const aclIntArray* actualSeqQLenOptional = nullptr;
    const aclIntArray* actualSeqKvLenOptional = nullptr;
    const aclTensor* attenMaskOptional = nullptr;
    double scaleValue;
    int64_t selectedBlockSize;
    int64_t selectedBlockCount;
    int64_t headNum;
    char* inputLayout;
    int64_t sparseMode;
    const aclTensor* dqOut = nullptr;
    const aclTensor* dkOut = nullptr;
    const aclTensor* dvOut = nullptr;
};

static aclnnStatus CheckParams(const NsaSelectedAttentionGradParams& params)
{
    CHECK_COND(params.query != nullptr, ACLNN_ERR_PARAM_NULLPTR, "query must not be nullptr.");
    CHECK_COND(params.key != nullptr, ACLNN_ERR_PARAM_NULLPTR, "key must not be nullptr.");
    CHECK_COND(params.value != nullptr, ACLNN_ERR_PARAM_NULLPTR, "value must not be nullptr.");
    CHECK_COND(params.attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "attentionOut must not be nullptr.");
    CHECK_COND(params.attentionOutGrad != nullptr, ACLNN_ERR_PARAM_NULLPTR, "attentionOutGrad must not be nullptr.");
    CHECK_COND(params.softmaxMax != nullptr, ACLNN_ERR_PARAM_NULLPTR, "softmaxMax must not be nullptr.");
    CHECK_COND(params.softmaxSum != nullptr, ACLNN_ERR_PARAM_NULLPTR, "softmaxSum must not be nullptr.");
    CHECK_COND(params.topkIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR, "topkIndices must not be nullptr.");
    CHECK_COND(params.dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dqOut must not be nullptr.");
    CHECK_COND(params.dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dkOut must not be nullptr.");
    CHECK_COND(params.dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dvOut must not be nullptr.");

    auto query_dtype = params.query->GetDataType();
    CHECK_COND(query_dtype == ACL_FLOAT16 || query_dtype == ACL_BF16, ACLNN_ERR_PARAM_INVALID,
               "The dtype of query is not support.");
    CHECK_COND(query_dtype == params.key->GetDataType(), ACLNN_ERR_PARAM_INVALID, "key dtype error.");
    CHECK_COND(query_dtype == params.value->GetDataType(), ACLNN_ERR_PARAM_INVALID, "value dtype error.");
    CHECK_COND(query_dtype == params.attentionOut->GetDataType(), ACLNN_ERR_PARAM_INVALID, "attentionOut dtype error.");
    CHECK_COND(query_dtype == params.attentionOutGrad->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "attentionOutGrad dtype error.");
    CHECK_COND(query_dtype == params.dqOut->GetDataType(), ACLNN_ERR_PARAM_INVALID, "dqOut dtype error.");
    CHECK_COND(query_dtype == params.dkOut->GetDataType(), ACLNN_ERR_PARAM_INVALID, "dkOut dtype error.");
    CHECK_COND(query_dtype == params.dvOut->GetDataType(), ACLNN_ERR_PARAM_INVALID, "dvOut dtype error.");

    auto softmaxMax_dtype = params.softmaxMax->GetDataType();
    auto softmaxSum_dtype = params.softmaxSum->GetDataType();
    auto topkIndices_dtype = params.topkIndices->GetDataType();
    CHECK_COND(softmaxMax_dtype == ACL_FLOAT, ACLNN_ERR_PARAM_INVALID, "softmaxMax dtype error.");
    CHECK_COND(softmaxSum_dtype == ACL_FLOAT, ACLNN_ERR_PARAM_INVALID, "softmaxSum dtype error.");
    CHECK_COND(topkIndices_dtype == ACL_INT32, ACLNN_ERR_PARAM_INVALID, "topkIndices dtype error.");

    return ACLNN_SUCCESS;
}

static bool CheckViewShape(const aclTensor* input1, const aclTensor* input2)
{
    int64_t* viewDims1 = nullptr;
    uint64_t viewDimsNum1 = 0;
    auto ret = aclGetViewShape(input1, &viewDims1, &viewDimsNum1);

    int64_t* viewDims2 = nullptr;
    uint64_t viewDimsNum2 = 0;
    ret = aclGetViewShape(input2, &viewDims2, &viewDimsNum2);

    if (viewDimsNum1 != viewDimsNum2) {
        return false;
    }

    for (uint64_t i = 0; i < viewDimsNum1; i++) {
        if (viewDims1[i] != viewDims2[i]) {
            return false;
        }
    }

    return true;
}
static void CheckFormatValid(const NsaSelectedAttentionGradParams& params)
{
    // 检查输入张量的格式是否为 ND 格式
    if (params.query != nullptr) {
        op::Format format = params.query->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of query gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.key != nullptr) {
        op::Format format = params.key->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of key gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.value != nullptr) {
        op::Format format = params.value->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of value gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.attentionOut != nullptr) {
        op::Format format = params.attentionOut->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of attentionOut gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.softmaxMax != nullptr) {
        op::Format format = params.softmaxMax->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of softmaxMax gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.softmaxSum != nullptr) {
        op::Format format = params.softmaxSum->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of softmaxSum gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.topkIndices != nullptr) {
        op::Format format = params.topkIndices->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of topkIndices gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.softmaxMax != nullptr) {
        op::Format format = params.softmaxMax->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of softmaxMax gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.softmaxSum != nullptr) {
        op::Format format = params.softmaxSum->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of softmaxSum gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.topkIndices != nullptr) {
        op::Format format = params.topkIndices->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of topkIndices gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }
}

aclnnStatus ContiguousAndNsaSelectedAttentionGrad(const NsaSelectedAttentionGradParams& params, aclOpExecutor* executor,
                                                  const aclTensor* dqOut, const aclTensor* dkOut,
                                                  const aclTensor* dvOut)
{
    auto queryContiguous = l0op::Contiguous(params.query, executor);
    CHECK_RET(queryContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto keyContiguous = l0op::Contiguous(params.key, executor);
    CHECK_RET(keyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto valueContiguous = l0op::Contiguous(params.value, executor);
    CHECK_RET(valueContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto attentionOutContiguous = l0op::Contiguous(params.attentionOut, executor);
    CHECK_RET(attentionOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto attentionOutGradContiguous = l0op::Contiguous(params.attentionOutGrad, executor);
    CHECK_RET(attentionOutGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto softmaxMaxContiguous = l0op::Contiguous(params.softmaxMax, executor);
    CHECK_RET(softmaxMaxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto softmaxSumContiguous = l0op::Contiguous(params.softmaxSum, executor);
    CHECK_RET(softmaxSumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto topkIndicesContiguous = l0op::Contiguous(params.topkIndices, executor);
    CHECK_RET(topkIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* attenMaskOptionalContiguous = nullptr;
    if (params.attenMaskOptional != nullptr) {
        attenMaskOptionalContiguous = l0op::Contiguous(params.attenMaskOptional, executor);
        CHECK_RET(attenMaskOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    string inputLayoutStr = op::ToString(params.inputLayout).GetString();
    CHECK_RET(inputLayoutStr == "TND", ACLNN_ERR_INNER_NULLPTR);

    // call l0 interface
    auto result = l0op::NsaSelectedAttentionGrad(
        queryContiguous, keyContiguous, valueContiguous, attentionOutContiguous, attentionOutGradContiguous,
        softmaxMaxContiguous, softmaxSumContiguous, topkIndicesContiguous, params.actualSeqQLenOptional,
        params.actualSeqKvLenOptional, attenMaskOptionalContiguous, params.scaleValue, params.selectedBlockCount,
        params.selectedBlockSize, params.headNum, inputLayoutStr.c_str(), params.sparseMode, executor);
    // convert output tensor to contiguous tensor
    CHECK_RET(result[0] != nullptr && result[1] != nullptr && result[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_COND(CheckViewShape(result[0], dqOut) != false, ACLNN_ERR_PARAM_INVALID, "The dq Tensor is Invaild");
    CHECK_COND(CheckViewShape(result[1], dkOut) != false, ACLNN_ERR_PARAM_INVALID, "The dk Tensor is Invaild");
    CHECK_COND(CheckViewShape(result[2], dvOut) != false, ACLNN_ERR_PARAM_INVALID, "The dv Tensor is Invaild");

    auto viewCopyResult = l0op::ViewCopy(result[0], dqOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    viewCopyResult = l0op::ViewCopy(result[1], dkOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    viewCopyResult = l0op::ViewCopy(result[2], dvOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaSelectedAttentionGradGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attentionOut,
    const aclTensor* attentionOutGrad, const aclTensor* softmaxMax, const aclTensor* softmaxSum,
    const aclTensor* topkIndices, const aclIntArray* actualSeqQLenOptional, const aclIntArray* actualSeqKvLenOptional,
    const aclTensor* attenMaskOptional, double scaleValue, int64_t selectedBlockSize, int64_t selectedBlockCount,
    int64_t headNum, char* inputLayout, int64_t sparseMode, const aclTensor* dqOut, const aclTensor* dkOut,
    const aclTensor* dvOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNsaSelectedAttentionGrad,
                   DFX_IN(query, key, value, attentionOut, attentionOutGrad, softmaxMax, softmaxSum, topkIndices,
                          actualSeqQLenOptional, actualSeqKvLenOptional, attenMaskOptional, scaleValue,
                          selectedBlockSize, selectedBlockCount, headNum, inputLayout, sparseMode),
                   DFX_OUT(dqOut, dkOut, dvOut));
    NsaSelectedAttentionGradParams params{query,
                                          key,
                                          value,
                                          attentionOut,
                                          attentionOutGrad,
                                          softmaxMax,
                                          softmaxSum,
                                          topkIndices,
                                          actualSeqQLenOptional,
                                          actualSeqKvLenOptional,
                                          attenMaskOptional,
                                          scaleValue,
                                          selectedBlockSize,
                                          selectedBlockCount,
                                          headNum,
                                          inputLayout,
                                          sparseMode,
                                          dqOut,
                                          dkOut,
                                          dvOut};
    // check params
    aclnnStatus ret = CheckParams(params);
    CHECK_COND(ret == ACLNN_SUCCESS, ret, "aclnnNsaSelectedAttentionGradGetWorkspaceSize checkParams failed.");
    CheckFormatValid(params);
    // create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (query->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // call l0 interface
    ret = ContiguousAndNsaSelectedAttentionGrad(params, uniqueExecutor.get(), dqOut, dkOut, dvOut);
    CHECK_COND(ret == ACLNN_SUCCESS, ret, "ContiguousAndNsaSelectedAttentionGrad failed.");
    // get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaSelectedAttentionGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNsaSelectedAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
