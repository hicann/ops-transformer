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
 * \file aclnn_sparse_flash_mla_softmax_l1_norm_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_flash_mla_softmax_l1_norm_metadata.h"
#include "l0_sparse_flash_mla_softmax_l1_norm_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor *metadata)
{
    CHECK_RET(metadata != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensKOptional, const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional, const aclTensor *cmpResidualKOptional, const aclTensor *topkLengthOptional,
    int64_t batchSize, int64_t maxSeqLenQ, int64_t maxSeqLenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim,
    int64_t topk, int64_t cmpRatio, int64_t maskMode, char *layoutQ, char *layoutK, const aclTensor *metadata,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSparseFlashMlaSoftmaxL1NormMetadata,
                   DFX_IN(cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional,
                          cmpResidualKOptional, topkLengthOptional, batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ,
                          numHeadsK, headDim, topk, cmpRatio, maskMode, layoutQ, layoutK),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    std::string socVersionStr = npuInfo.GetSocLongVersion();
    const char *socVersion = socVersionStr.c_str();

    const aclTensor *cuSeqLensQContiguous = nullptr;
    if (cuSeqLensQOptional != nullptr) {
        cuSeqLensQContiguous = l0op::Contiguous(cuSeqLensQOptional, uniqueExecutor.get());
        CHECK_RET(cuSeqLensQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cuSeqLensKContiguous = nullptr;
    if (cuSeqLensKOptional != nullptr) {
        cuSeqLensKContiguous = l0op::Contiguous(cuSeqLensKOptional, uniqueExecutor.get());
        CHECK_RET(cuSeqLensKContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *seqUsedQContiguous = nullptr;
    if (seqUsedQOptional != nullptr) {
        seqUsedQContiguous = l0op::Contiguous(seqUsedQOptional, uniqueExecutor.get());
        CHECK_RET(seqUsedQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *seqUsedKContiguous = nullptr;
    if (seqUsedKOptional != nullptr) {
        seqUsedKContiguous = l0op::Contiguous(seqUsedKOptional, uniqueExecutor.get());
        CHECK_RET(seqUsedKContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cmpResidualKContiguous = nullptr;
    if (cmpResidualKOptional != nullptr) {
        cmpResidualKContiguous = l0op::Contiguous(cmpResidualKOptional, uniqueExecutor.get());
        CHECK_RET(cmpResidualKContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *topkLengthContiguous = nullptr;
    if (topkLengthOptional != nullptr) {
        topkLengthContiguous = l0op::Contiguous(topkLengthOptional, uniqueExecutor.get());
        CHECK_RET(topkLengthContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto output = l0op::SparseFlashMlaSoftmaxL1NormMetadata(
        cuSeqLensQContiguous, cuSeqLensKContiguous, seqUsedQContiguous, seqUsedKContiguous, cmpResidualKContiguous,
        topkLengthContiguous, batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim, topk, cmpRatio,
        maskMode, layoutQ, layoutK, socVersion, aicCoreNum, metadata, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormMetadata(void *workspace,
                                                                                            uint64_t workspaceSize,
                                                                                            aclOpExecutor *executor,
                                                                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseFlashMlaSoftmaxL1NormMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
