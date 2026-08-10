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
 * \file aclnn_quant_lightning_indexer_metadata.cpp
 * \brief
 */

#include "aclnn_quant_lightning_indexer_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "l0_quant_lightning_indexer_metadata.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "../quant_lightning_indexer_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnQuantLightningIndexerMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKeyOptional, int64_t numHeadsQ,
    int64_t numHeadsK, int64_t headDim, int64_t queryQuantMode, int64_t keyQuantMode, int64_t batchSize,
    int64_t maxSeqlenQ, int64_t maxSeqlenK, char* layoutQueryOptional, char* layoutKeyOptional,
    int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t cmpRatio,
    const aclTensor* metadata, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantLightningIndexerMetadata,
                   DFX_IN(actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, numHeadsQ, numHeadsK, headDim,
                          queryQuantMode, keyQuantMode, batchSize, maxSeqlenQ, maxSeqlenK, layoutQueryOptional,
                          layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, cmpRatio),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char* socVersion = npuInfo.GetSocLongVersion().c_str();

    auto ret = ParamsCheck(actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, numHeadsQ, numHeadsK, headDim,
                           queryQuantMode, keyQuantMode, batchSize, maxSeqlenQ, maxSeqlenK, layoutQueryOptional,
                           layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, cmpRatio, socVersion,
                           metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto actualSeqLengthsQueryOptionalContiguous =
        l0op::Contiguous(actualSeqLengthsQueryOptional, uniqueExecutor.get());
    if (actualSeqLengthsQueryOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "actual_seq_lengths_query contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto actualSeqLengthsKeyOptionalContiguous = l0op::Contiguous(actualSeqLengthsKeyOptional, uniqueExecutor.get());
    if (actualSeqLengthsKeyOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "actual_seq_lengths_key contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto output = l0op::QuantLightningIndexerMetadata(
        actualSeqLengthsQueryOptionalContiguous, actualSeqLengthsKeyOptionalContiguous, aicCoreNum, aivCoreNum,
        socVersion, numHeadsQ, numHeadsK, headDim, queryQuantMode, keyQuantMode, batchSize, maxSeqlenQ, maxSeqlenK,
        layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, cmpRatio, metadata,
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnQuantLightningIndexerMetadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantLightningIndexerMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
