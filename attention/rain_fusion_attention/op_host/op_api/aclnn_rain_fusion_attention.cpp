/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_rain_fusion_attention.h"

#include "rain_fusion_attention.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include <acl/acl.h>
#include <algorithm>
#include <unordered_map>
#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static bool CheckDataType(const aclTensor *query, const aclTensor *key, const aclTensor *value)
{
    const DataType qDtype = query->GetDataType();
    const DataType kDtype = key->GetDataType();
    const DataType vDtype = value->GetDataType();

    static const std::unordered_map<DataType, std::vector<DataType>> validKvType = {
        {DataType::DT_FLOAT16, {DataType::DT_FLOAT16}},
        {DataType::DT_BF16, {DataType::DT_BF16}},
    };

    auto iter = validKvType.find(qDtype);
    if (iter == validKvType.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported query datatype %d.", static_cast<int>(qDtype));
        return false;
    }

    if (std::find(iter->second.begin(), iter->second.end(), kDtype) == iter->second.end() ||
        std::find(iter->second.begin(), iter->second.end(), vDtype) == iter->second.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Key/Value datatype mismatch with query.");
        return false;
    }

    return true;
}

static aclnnStatus CheckMandatoryTensors(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                         const aclTensor *selectIdx, const aclTensor *selectNumIdx)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(selectIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(selectNumIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseBlockShape(const aclIntArray *blockShape)
{
    if (blockShape == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "blockShape is null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    uint64_t size = blockShape->Size();
    if (size < 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShape must contain at least two elements [x, y].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t *data = blockShape->GetData();
    if (data == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShape data is null.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (data[0] <= 0 || data[1] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShape values must be positive, got [%ld, %ld].", data[0], data[1]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateParams(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                  const aclTensor *selectIdx, const aclTensor *selectNumIdx, char *qInputLayout,
                                  char *kvInputLayout, const aclIntArray *blockShape)
{
    CHECK_RET(CheckMandatoryTensors(query, key, value, selectIdx, selectNumIdx) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    if (!CheckDataType(query, key, value)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qInputLayout == nullptr || kvInputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input layout strings are null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    std::string qLayout(qInputLayout);
    std::string kvLayout(kvInputLayout);

    // 验证Q layout
    if (qLayout != "TND" && qLayout != "BNSD") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qInputLayout only supports TND or BNSD, got %s.", qLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 验证KV layout
    if (kvLayout != "TND" && kvLayout != "BNSD") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvInputLayout only supports TND or BNSD, got %s.", kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 验证Q和KV格式一致性：如果其中一个是BNSD，另一个也必须是BNSD
    bool qIsBNSD = (qLayout == "BNSD");
    bool kvIsBNSD = (kvLayout == "BNSD");
    if (qIsBNSD != kvIsBNSD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Q and KV layouts must match: if one is BNSD, the other must also be BNSD. "
                "Q layout: %s, KV layout: %s",
                qLayout.c_str(), kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ParseBlockShape(blockShape);
}

static aclnnStatus MakeContiguous(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                  const aclTensor *&attenMaskOptional, const aclTensor *&blockTableOptional,
                                  const aclTensor *&selectIdx, const aclTensor *&selectNumIdx, aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    selectIdx = l0op::Contiguous(selectIdx, executor);
    CHECK_RET(selectIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    selectNumIdx = l0op::Contiguous(selectNumIdx, executor);
    CHECK_RET(selectNumIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (attenMaskOptional != nullptr) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (blockTableOptional != nullptr) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateAdditionalParams(int64_t innerPrecise, const aclTensor *attentionOut,
                                            uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (innerPrecise != 0 && innerPrecise != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "innerPrecise must be 0 (float32 softmax) or 1 (fp16 softmax), got %ld.",
                innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static string ConvertLayoutString(char *layoutStr)
{
    return op::ToString(layoutStr).GetString();
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnRainFusionAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *selectIdx,
    const aclTensor *selectNumIdx, const aclIntArray *blockShape, const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *blockTableOptional, char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads,
    int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, const aclTensor *attentionOut,
    const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret =
        ValidateParams(query, key, value, selectIdx, selectNumIdx, qInputLayout, kvInputLayout, blockShape);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = ValidateAdditionalParams(innerPrecise, attentionOut, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    L2_DFX_PHASE_1(
        aclnnRainFusionAttention,
        DFX_IN(query, key, value, selectIdx, selectNumIdx, blockShape, attenMaskOptional, actualSeqLengthsOptional,
               actualSeqLengthsKvOptional, blockTableOptional, qInputLayout, kvInputLayout, numKeyValueHeads, maskType,
               scaleValue, innerPrecise, blockSize, attentionOut, softmaxLseOptional),
        DFX_OUT(attentionOut, softmaxLseOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    ret =
        MakeContiguous(query, key, value, attenMaskOptional, blockTableOptional, selectIdx, selectNumIdx, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    string qInputLayoutStr = ConvertLayoutString(qInputLayout);
    string kvInputLayoutStr = ConvertLayoutString(kvInputLayout);

    auto outputs = l0op::RainFusionAttention(query, key, value, selectIdx, selectNumIdx, blockShape, attenMaskOptional,
                                             actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional,
                                             qInputLayoutStr.c_str(), kvInputLayoutStr.c_str(), numKeyValueHeads,
                                             maskType, scaleValue, innerPrecise, blockSize, executorImpl);
    if (outputs[0] == nullptr || outputs[1] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RainFusionAttention returned nullptr outputs.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(outputs[0], attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnRainFusionAttention(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRainFusionAttention);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
