/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_generic_block_sparse_attention_grad.cpp
 * \brief L2 aclnn API for GenericBlockSparseAttentionGrad.
 */

#include "aclnn_generic_block_sparse_attention_grad.h"
#include "generic_block_sparse_attention_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr int64_t GSAG_MAX_HEAD_NUM = 128;
static constexpr int64_t GSAG_SUPPORTED_MASK_TYPE = 1;

inline bool TensorOk(const aclTensor *t)
{
    return t != nullptr && t->GetViewShape().GetDimNum() > 0;
}

int64_t GetHeadNumFromQuery(const aclTensor *query, const char *layout)
{
    const auto &shape = query->GetViewShape();
    if (strcmp(layout, "TND") == 0) {
        return shape.GetDimNum() >= 2 ? shape.GetDim(1) : -1;
    }
    if (strcmp(layout, "BNSD") == 0) {
        return shape.GetDimNum() >= 2 ? shape.GetDim(1) : -1;
    }
    if (strcmp(layout, "BSND") == 0) {
        return shape.GetDimNum() >= 3 ? shape.GetDim(2) : -1;
    }
    return -1;
}

int64_t GetHeadNumFromKey(const aclTensor *key, const char *layout)
{
    const auto &shape = key->GetViewShape();
    if (strcmp(layout, "TND") == 0) {
        return shape.GetDimNum() >= 2 ? shape.GetDim(1) : -1;
    }
    if (strcmp(layout, "BNSD") == 0) {
        return shape.GetDimNum() >= 2 ? shape.GetDim(1) : -1;
    }
    if (strcmp(layout, "BSND") == 0) {
        return shape.GetDimNum() >= 3 ? shape.GetDim(2) : -1;
    }
    return -1;
}

aclnnStatus Validate(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout,
                     const aclTensor *out, const aclTensor *lse, const aclTensor *sparseBlockIdx,
                     const aclTensor *sparseBlockCount, const aclTensor *metadata, const aclTensor *attenMaskOptional,
                     const aclIntArray *blockShape, int64_t isPackedGQA, char *layoutQ, char *layoutKv,
                     int64_t maskType, int64_t winLeft, int64_t winRight, const aclTensor *dQuery,
                     const aclTensor *dKey, const aclTensor *dValue)
{
    CHECK_RET(TensorOk(query) && TensorOk(key) && TensorOk(value) && TensorOk(dout) && TensorOk(out) && TensorOk(lse),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(TensorOk(sparseBlockIdx) && TensorOk(sparseBlockCount) && TensorOk(metadata), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(TensorOk(dQuery) && TensorOk(dKey) && TensorOk(dValue), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(layoutQ != nullptr && layoutKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(strcmp(layoutQ, layoutKv) == 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(isPackedGQA == 1, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(maskType == GSAG_SUPPORTED_MASK_TYPE, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(winLeft == -1 && winRight == -1, ACLNN_ERR_PARAM_INVALID);
    // atten_mask is not supported yet; must be nullptr (causal uses maskType only).
    CHECK_RET(attenMaskOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
    if (blockShape != nullptr) {
        CHECK_RET(blockShape->Size() >= 2, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET((*blockShape)[0] == 1, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET((*blockShape)[1] >= 128 && ((*blockShape)[1] % 64 == 0), ACLNN_ERR_PARAM_INVALID);
    }
    const int64_t qHeadNum = GetHeadNumFromQuery(query, layoutQ);
    const int64_t kvHeadNum = GetHeadNumFromKey(key, layoutKv);
    CHECK_RET(qHeadNum > 0 && qHeadNum <= GSAG_MAX_HEAD_NUM, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(kvHeadNum > 0 && kvHeadNum <= GSAG_MAX_HEAD_NUM, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(qHeadNum % kvHeadNum == 0, ACLNN_ERR_PARAM_INVALID);
    DataType qDtype = query->GetDataType();
    CHECK_RET(qDtype == ACL_FLOAT16 || qDtype == ACL_BF16, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(key->GetDataType() == qDtype && value->GetDataType() == qDtype && dout->GetDataType() == qDtype &&
                  out->GetDataType() == qDtype,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(lse->GetDataType() == ACL_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(sparseBlockIdx->GetDataType() == ACL_INT32 && sparseBlockCount->GetDataType() == ACL_INT32,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(metadata->GetDataType() == ACL_INT64, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout, const aclTensor *out,
    const aclTensor *lse, const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *metadata,
    const aclTensor *attenMaskOptional, const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, const aclIntArray *blockShape,
    int64_t isPackedGQA, char *layoutQ, char *layoutKv, double scaleValue, int64_t maskType, int64_t softmaxPrecision,
    int64_t winLeft, int64_t winRight, aclTensor *dQuery, aclTensor *dKey, aclTensor *dValue, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(
        aclnnGenericBlockSparseAttentionGrad,
        DFX_IN(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadata, attenMaskOptional,
               cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, blockShape,
               isPackedGQA, layoutQ, layoutKv, scaleValue, maskType, softmaxPrecision, winLeft, winRight),
        DFX_OUT(dQuery, dKey, dValue));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret =
        Validate(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadata, attenMaskOptional,
                 blockShape, isPackedGQA, layoutQ, layoutKv, maskType, winLeft, winRight, dQuery, dKey, dValue);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto queryC = l0op::Contiguous(query, uniqueExecutor.get());
    auto keyC = l0op::Contiguous(key, uniqueExecutor.get());
    auto valueC = l0op::Contiguous(value, uniqueExecutor.get());
    auto doutC = l0op::Contiguous(dout, uniqueExecutor.get());
    auto outC = l0op::Contiguous(out, uniqueExecutor.get());
    auto lseC = l0op::Contiguous(lse, uniqueExecutor.get());
    auto idxC = l0op::Contiguous(sparseBlockIdx, uniqueExecutor.get());
    auto cntC = l0op::Contiguous(sparseBlockCount, uniqueExecutor.get());
    auto metaC = l0op::Contiguous(metadata, uniqueExecutor.get());
    CHECK_RET(queryC && keyC && valueC && doutC && outC && lseC && idxC && cntC && metaC, ACLNN_ERR_INNER_NULLPTR);

    // attenMaskOptional already required nullptr in Validate; keep placeholder for OP_INPUT slot.
    const aclTensor *attenC = nullptr;
    const aclTensor *cuQC = nullptr;
    if (cuSeqLengthsQOptional != nullptr) {
        cuQC = l0op::Contiguous(cuSeqLengthsQOptional, uniqueExecutor.get());
        CHECK_RET(cuQC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cuKvC = nullptr;
    if (cuSeqLengthsKvOptional != nullptr) {
        cuKvC = l0op::Contiguous(cuSeqLengthsKvOptional, uniqueExecutor.get());
        CHECK_RET(cuKvC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedQC = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQC = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        CHECK_RET(sequsedQC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedKvC = nullptr;
    if (sequsedKvOptional != nullptr) {
        sequsedKvC = l0op::Contiguous(sequsedKvOptional, uniqueExecutor.get());
        CHECK_RET(sequsedKvC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outs = l0op::GenericBlockSparseAttentionGrad(queryC, keyC, valueC, doutC, outC, lseC, idxC, cntC, metaC,
                                                      attenC, cuQC, cuKvC, sequsedQC, sequsedKvC, blockShape,
                                                      isPackedGQA, layoutQ, layoutKv, scaleValue, maskType,
                                                      softmaxPrecision, winLeft, winRight, uniqueExecutor.get());
    CHECK_RET(outs[0] != nullptr && outs[1] != nullptr && outs[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dqView = l0op::ViewCopy(outs[0], dQuery, uniqueExecutor.get());
    auto dkView = l0op::ViewCopy(outs[1], dKey, uniqueExecutor.get());
    auto dvView = l0op::ViewCopy(outs[2], dValue, uniqueExecutor.get());
    CHECK_RET(dqView != nullptr && dkView != nullptr && dvView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGrad(void *workspace,
                                                                                        uint64_t workspaceSize,
                                                                                        aclOpExecutor *executor,
                                                                                        const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGenericBlockSparseAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
