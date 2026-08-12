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
 * \file aclnn_sparse_flash_attention_antiquant_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_flash_attention_antiquant_metadata.h"
#include "l0_sparse_flash_attention_antiquant_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor *actualSeqLengthsQueryOptional,
                               const aclTensor *actualSeqLengthsKvOptional,
                               const aclTensor *sparseSeqLengthsKvOptional,
                               int64_t batchSize,
                               int64_t querySeqSize,
                               int64_t queryHeadNum,
                               int64_t kvSeqSize,
                               int64_t kvHeadNum,
                               int64_t headDim,
                               int64_t topkSize,
                               int64_t sparseBlockSize,
                               char *layoutQueryOptional,
                               char *layoutKvOptional,
                               int64_t sparseMode,
                               int64_t attentionMode,
                               int64_t ropeHeadDim,
                               int64_t sparseSharedSize,
                               const aclTensor *metaData)
{
    if (batchSize < 0 || querySeqSize < 0 || queryHeadNum < 0 ||
        kvSeqSize < 0 || kvHeadNum < 0 || headDim < 0 ||
        sparseBlockSize < 0 || sparseSharedSize < 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvQuantSparseFlashAttentionMetadataGetWorkspaceSize(
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *sparseSeqLengthsKvOptional,
    int64_t batchSize,
    int64_t querySeqSize,
    int64_t queryHeadNum,
    int64_t kvSeqSize,
    int64_t kvHeadNum,
    int64_t headDim,
    int64_t topkSize,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    int64_t attentionMode,
    int64_t ropeHeadDim,
    int64_t sparseSharedSize,
    const aclTensor *metaData,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnKvQuantSparseFlashAttentionMetadata,
        DFX_IN(actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
               sparseSeqLengthsKvOptional, batchSize, querySeqSize, queryHeadNum,
               kvSeqSize, kvHeadNum, headDim, topkSize, sparseBlockSize,
               layoutQueryOptional, layoutKvOptional,
               sparseMode, attentionMode, ropeHeadDim, sparseSharedSize),
        DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(
        actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
        sparseSeqLengthsKvOptional, batchSize, querySeqSize, queryHeadNum,
        kvSeqSize, kvHeadNum, headDim, topkSize, sparseBlockSize,
        layoutQueryOptional, layoutKvOptional, sparseMode,
        attentionMode, ropeHeadDim, sparseSharedSize, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();

    std::string socVersionStr = npuInfo.GetSocLongVersion();
    const char *socVersionOptional = socVersionStr.c_str();

    auto output = l0op::KvQuantSparseFlashAttentionMetadata(
        actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
        sparseSeqLengthsKvOptional, batchSize, querySeqSize, queryHeadNum,
        kvSeqSize, kvHeadNum, headDim, topkSize, sparseBlockSize, aicCoreNum,
        aivCoreNum, layoutQueryOptional, layoutKvOptional, sparseMode,
        attentionMode, ropeHeadDim, sparseSharedSize, socVersionOptional, metaData,
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnKvQuantSparseFlashAttentionMetadata(void *workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKvQuantSparseFlashAttentionMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
