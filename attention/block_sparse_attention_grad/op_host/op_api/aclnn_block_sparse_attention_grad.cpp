/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_block_sparse_attention_grad.h"
#include "aclnn_kernels/transpose.h"
#include "block_sparse_attention_grad.h"
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
struct BSAGParams {
    const aclTensor *query = nullptr;
    const aclTensor *key = nullptr;
    const aclTensor *value = nullptr;
    const aclTensor *attentionOut = nullptr;
    const aclTensor *attentionOutGrad = nullptr;
    const aclTensor *softmaxLse = nullptr;
    const aclTensor *blockSparseMaskOptional = nullptr;
    const aclTensor *attenMaskOptional = nullptr;
    const aclIntArray *actualSeqLengthsOptional = nullptr;
    const aclIntArray *actualSeqLengthsKvOptional = nullptr;
    char *qInputLayout;
    char *kvInputLayout;
    const aclIntArray *blockShapeOptional;
    int64_t numKeyValueHeads;
    int64_t maskType;
    int64_t preTokens;
    int64_t nextTokens;
    const aclTensor *dqOut = nullptr;
    const aclTensor *dkOut = nullptr;
    const aclTensor *dvOut = nullptr;
};


static aclnnStatus CheckMandatoryTensors(const aclTensor *dout,
                                         const aclTensor *query,
                                         const aclTensor *key,
                                         const aclTensor *value,
                                         const aclTensor *attentionOut,
                                         const aclTensor *softmaxLse,
                                         const aclTensor *blockSparseMaskOptional)
{
    CHECK_RET(dout != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseBlockShape(const aclIntArray *blockShapeOptional)
{
    if (blockShapeOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "blockShapeOptional is null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    uint64_t size = blockShapeOptional->Size();
    if (size < 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional must contain at least two elements [x, y].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t *data = blockShapeOptional->GetData();
    if (data == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional data is null.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (data[0] <= 0 || data[1] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional values must be positive, got [%ld, %ld].", data[0], data[1]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}
static aclnnStatus ValidateParams(const BSAGParams& params)
{
    CHECK_COND(params.qInputLayout != nullptr && params.kvInputLayout != nullptr,
        ACLNN_ERR_PARAM_INVALID, "qInputLayout or kvInputLayout is nullptr.");
    std::string qLayout(params.qInputLayout);
    std::string kvLayout(params.kvInputLayout);
    CHECK_RET(CheckMandatoryTensors(params.attentionOutGrad, params.query, params.key, params.value,
                                    params.attentionOut, params.softmaxLse, params.blockSparseMaskOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    // dtype检查
    DataType qDtype = params.query->GetDataType();
    DataType kDtype = params.key->GetDataType();
    DataType vDtype = params.value->GetDataType();
    DataType attnOutDtype = params.attentionOut->GetDataType();
    DataType gradDtype = params.attentionOutGrad->GetDataType();   // dout
    DataType lseDtype = params.softmaxLse->GetDataType();
    DataType maskDtype = params.blockSparseMaskOptional->GetDataType();
    DataType dqDtype = params.dqOut->GetDataType();
    DataType dkDtype = params.dkOut->GetDataType();
    DataType dvDtype = params.dvOut->GetDataType();
    CHECK_COND(qDtype == ACL_FLOAT16 || qDtype == ACL_BF16, ACLNN_ERR_PARAM_INVALID, "The dtype of query is not support.");
    CHECK_COND(qDtype == kDtype, ACLNN_ERR_PARAM_INVALID, "key dtype error.");
    CHECK_COND(qDtype == vDtype, ACLNN_ERR_PARAM_INVALID, "value dtype error.");
    CHECK_COND(attnOutDtype == qDtype, ACLNN_ERR_PARAM_INVALID, "attentionOut dtype must match query dtype.");
    CHECK_COND(gradDtype == qDtype, ACLNN_ERR_PARAM_INVALID, "attentionOutGrad dtype must match query dtype.");
    CHECK_COND(lseDtype == ACL_FLOAT, ACLNN_ERR_PARAM_INVALID, "softmaxLse dtype must be FLOAT32.");
    CHECK_COND(maskDtype == ACL_BOOL || maskDtype == ACL_UINT8, ACLNN_ERR_PARAM_INVALID, "blockSparseMask dtype must be BOOL or UINT8.");
    CHECK_COND(dqDtype == qDtype, ACLNN_ERR_PARAM_INVALID, "dqOut dtype must match query dtype.");
    CHECK_COND(dkDtype == qDtype, ACLNN_ERR_PARAM_INVALID, "dkOut dtype must match query dtype.");
    CHECK_COND(dvDtype == qDtype, ACLNN_ERR_PARAM_INVALID, "dvOut dtype must match query dtype.");
    // format检查
    if (params.query->GetStorageFormat() != ge::FORMAT_ND ||
        params.key->GetStorageFormat() != ge::FORMAT_ND ||
        params.value->GetStorageFormat() != ge::FORMAT_ND ||
        params.attentionOut->GetStorageFormat() != ge::FORMAT_ND ||
        params.attentionOutGrad->GetStorageFormat() != ge::FORMAT_ND ||
        params.softmaxLse->GetStorageFormat() != ge::FORMAT_ND ||
        params.blockSparseMaskOptional->GetStorageFormat() != ge::FORMAT_ND ||
        params.dqOut->GetStorageFormat() != ge::FORMAT_ND ||
        params.dkOut->GetStorageFormat() != ge::FORMAT_ND ||
        params.dvOut->GetStorageFormat() != ge::FORMAT_ND) {
        OP_LOGW("Format of input is not ND, this format may lead to precision failure.");
    }
    // shape检查
    auto queryShape = params.query->GetViewShape();
    auto keyShape = params.key->GetViewShape();
    auto valueShape = params.value->GetViewShape();
    auto attnOutShape = params.attentionOut->GetViewShape();
    auto gradShape = params.attentionOutGrad->GetViewShape();
    auto lseShape = params.softmaxLse->GetViewShape();
    auto dqShape = params.dqOut->GetViewShape();
    auto dkShape = params.dkOut->GetViewShape();
    auto dvShape = params.dvOut->GetViewShape();
    if (qLayout == "TND") {
        // check head_dim
        CHECK_COND(queryShape.GetDim(2) == keyShape.GetDim(2), ACLNN_ERR_PARAM_INVALID, "The dim's 2 of query and key must be same.");
        CHECK_COND(queryShape.GetDim(2) == valueShape.GetDim(2), ACLNN_ERR_PARAM_INVALID, "The dim's 2 of query and value must be same.");
        CHECK_COND(queryShape.GetDim(2) == attnOutShape.GetDim(2), ACLNN_ERR_PARAM_INVALID, "The dim's 2 of query and attentionOut must be same.");
        CHECK_COND(queryShape.GetDim(2) == gradShape.GetDim(2), ACLNN_ERR_PARAM_INVALID, "The dim's 2 of query and attentionOutGrad must be same.");
        CHECK_COND(queryShape.GetDim(2) == dqShape.GetDim(2), ACLNN_ERR_PARAM_INVALID, "The dim's 2 of query and dqOut must be same.");
        CHECK_COND(queryShape.GetDim(2) == dkShape.GetDim(2), ACLNN_ERR_PARAM_INVALID, "The dim's 2 of query and dkOut must be same.");
        CHECK_COND(queryShape.GetDim(2) == dvShape.GetDim(2), ACLNN_ERR_PARAM_INVALID, "The dim's 2 of query and dvOut must be same.");
    } else {
        //check batch_num, head_dim
        CHECK_COND(queryShape.GetDim(0) == keyShape.GetDim(0), ACLNN_ERR_PARAM_INVALID, "The dim's 0 of query and key must be same.");
        CHECK_COND(queryShape.GetDim(0) == valueShape.GetDim(0), ACLNN_ERR_PARAM_INVALID, "The dim's 0 of query and value must be same.");
        CHECK_COND(queryShape.GetDim(0) == attnOutShape.GetDim(0), ACLNN_ERR_PARAM_INVALID, "The dim's 0 of query and attentionOut must be same.");
        CHECK_COND(queryShape.GetDim(0) == gradShape.GetDim(0), ACLNN_ERR_PARAM_INVALID, "The dim's 0 of query and attentionOutGrad must be same.");
        CHECK_COND(queryShape.GetDim(0) == lseShape.GetDim(0), ACLNN_ERR_PARAM_INVALID, "The dim's 0 of query and softmaxLse must be same.");
        CHECK_COND(queryShape.GetDim(0) == dqShape.GetDim(0), ACLNN_ERR_PARAM_INVALID, "The dim's 0 of query and dqOut must be same.");
        CHECK_COND(queryShape.GetDim(0) == dkShape.GetDim(0), ACLNN_ERR_PARAM_INVALID, "The dim's 0 of query and dkOut must be same.");
        CHECK_COND(queryShape.GetDim(0) == dvShape.GetDim(0), ACLNN_ERR_PARAM_INVALID, "The dim's 0 of query and dvOut must be same.");
        CHECK_COND(queryShape.GetDim(3) == keyShape.GetDim(3), ACLNN_ERR_PARAM_INVALID, "The dim's 3 of query and key must be same.");
        CHECK_COND(valueShape.GetDim(3) == attnOutShape.GetDim(3), ACLNN_ERR_PARAM_INVALID, "The dim's 3 of value and attentionOut must be same.");
        CHECK_COND(valueShape.GetDim(3) == gradShape.GetDim(3), ACLNN_ERR_PARAM_INVALID, "The dim's 3 of value and attentionOutGrad must be same.");
        CHECK_COND(queryShape.GetDim(3) == dqShape.GetDim(3), ACLNN_ERR_PARAM_INVALID, "The dim's 3 of query and dqOut must be same.");
        CHECK_COND(keyShape.GetDim(3) == dkShape.GetDim(3), ACLNN_ERR_PARAM_INVALID, "The dim's 3 of key and dkOut must be same.");
        CHECK_COND(valueShape.GetDim(3) == dvShape.GetDim(3), ACLNN_ERR_PARAM_INVALID, "The dim's 3 of value and dvOut must be same.");
    }

    if (params.attenMaskOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "attenMaskOptional currently only supports nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (params.maskType != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskType only supports 0, got %ld.", params.maskType);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (params.preTokens != 2147483647 || params.nextTokens != 2147483647) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "preTokens and nextTokens must be 2147483647, got [%ld,%ld].",
                params.preTokens, params.nextTokens);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (params.qInputLayout == nullptr || params.kvInputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input layout strings are null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (qLayout == "TND" && params.actualSeqLengthsOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, " actualSeqLengthsOptional  is mandatory when qInputLayout is TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kvLayout == "TND" && params.actualSeqLengthsKvOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, " actualSeqLengthsKvOptional  is mandatory when kvInputLayout is TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 验证Q和KV格式一致性：如果其中一个是BNSD，另一个也必须是BNSD
    bool qIsBNSD = (qLayout == "BNSD");
    bool kvIsBNSD = (kvLayout == "BNSD");
    if (qIsBNSD != kvIsBNSD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Q and KV layouts must match: if one is BNSD, the other must also be BNSD. "
                "Q layout: %s, KV layout: %s", qLayout.c_str(), kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ParseBlockShape(params.blockShapeOptional);
}

static aclnnStatus MakeContiguous(const aclTensor *&dout,
                                  const aclTensor *&query,
                                  const aclTensor *&key,
                                  const aclTensor *&value,
                                  const aclTensor *&attentionOut,
                                  const aclTensor *&softmaxLse,
                                  const aclTensor *&blockSparseMaskOptional,
                                  const aclTensor *&attenMaskOptional,
                                  aclOpExecutor *executor)
{
    dout = l0op::Contiguous(dout, executor);
    CHECK_RET(dout != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    attentionOut = l0op::Contiguous(attentionOut, executor);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    softmaxLse = l0op::Contiguous(softmaxLse, executor);
    CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    blockSparseMaskOptional = l0op::Contiguous(blockSparseMaskOptional, executor);
    CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (attenMaskOptional != nullptr) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static string ConvertLayoutString(char *layoutStr)
{
    return op::ToString(layoutStr).GetString();
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *dout,
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    char *qInputLayout,
    char *kvInputLayout,
    int64_t numKeyValueHeads,
    int64_t maskType,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    aclTensor *dq,
    aclTensor *dk,
    aclTensor *dv,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    BSAGParams bsag_params;
    bsag_params.query = query;
    bsag_params.key = key;
    bsag_params.value = value;
    bsag_params.attentionOut = attentionOut;
    bsag_params.attentionOutGrad = dout;
    bsag_params.softmaxLse = softmaxLse;
    bsag_params.blockSparseMaskOptional = blockSparseMaskOptional;
    bsag_params.attenMaskOptional = attenMaskOptional;
    bsag_params.actualSeqLengthsOptional = actualSeqLengthsOptional;
    bsag_params.actualSeqLengthsKvOptional = actualSeqLengthsKvOptional;
    bsag_params.qInputLayout = qInputLayout;
    bsag_params.kvInputLayout = kvInputLayout;
    bsag_params.blockShapeOptional = blockShapeOptional;
    bsag_params.numKeyValueHeads = numKeyValueHeads;
    bsag_params.maskType = maskType;
    bsag_params.preTokens = preTokens;
    bsag_params.nextTokens = nextTokens;
    bsag_params.dqOut = dq;
    bsag_params.dkOut = dk;
    bsag_params.dvOut = dv;
    aclnnStatus ret = ValidateParams(bsag_params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    L2_DFX_PHASE_1(aclnnBlockSparseAttentionGrad,
                   DFX_IN(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                          actualSeqLengthsOptional, actualSeqLengthsKvOptional, qInputLayout, qInputLayout, numKeyValueHeads,
                          maskType, scaleValue, preTokens, nextTokens),
                   DFX_OUT(dq, dk, dv));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (blockSparseMaskOptional != nullptr) {
            int64_t valuePerm[4] = {0, 1, 3, 2};
            auto perm = executorImpl->AllocIntArray(valuePerm, 4);
            blockSparseMaskOptional = l0op::Contiguous(blockSparseMaskOptional, executorImpl);
            blockSparseMaskOptional = l0op::Transpose(blockSparseMaskOptional, perm, executorImpl);
            CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    ret = MakeContiguous(dout, query, key, value, attentionOut, softmaxLse,blockSparseMaskOptional, attenMaskOptional, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    string qInputLayoutStr = ConvertLayoutString(qInputLayout);
    string kvInputLayoutStr = ConvertLayoutString(kvInputLayout);
    
    auto outputs = l0op::BlockSparseAttentionGrad(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, 
                                                  attenMaskOptional, blockShapeOptional,actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                                  const_cast<char*>(qInputLayoutStr.c_str()), 
                                                  const_cast<char*>(kvInputLayoutStr.c_str()), numKeyValueHeads,
                                                  maskType, scaleValue, preTokens, nextTokens, executorImpl);
    if (outputs[0] == nullptr || outputs[1] == nullptr || outputs[2] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BlockSparseAttentionGrad returned nullptr outputs.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult0 = l0op::ViewCopy(outputs[0], dq, executorImpl);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(outputs[1], dk, executorImpl);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult2 = l0op::ViewCopy(outputs[2], dv, executorImpl);
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockSparseAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
