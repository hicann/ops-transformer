/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_lightning_indexer_v3_metadata.h"
#include <string>
#include <limits>
#include "acl/acl_rt.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "quant_lightning_indexer_v3_metadata.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

namespace {
constexpr int64_t QUERY_HEAD_NUM_32 = 32;
constexpr int64_t QUERY_HEAD_NUM_64 = 64;
constexpr int64_t KEY_HEAD_NUM = 1;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t TOPK_MAX = 2048;
constexpr int64_t QUANT_MODE_INT8 = 2;
constexpr int64_t MASK_MODE_NONE = 0;
constexpr int64_t MASK_MODE_CAUSAL = 3;
constexpr int64_t CMP_RATIO_MAX = 128;
constexpr int64_t METADATA_SIZE = 1024;
constexpr int64_t CANDIDATE_TOPK_BLOCKS = 2048;
constexpr int64_t CANDIDATE_BLOCK_SIZE = 8;

bool IsInt32Tensor(const aclTensor* tensor)
{
    aclDataType dtype = ACL_DT_UNDEFINED;
    return tensor != nullptr && aclGetDataType(tensor, &dtype) == ACL_SUCCESS && dtype == ACL_INT32 &&
           op::IsContiguous(tensor);
}

bool IsSequenceTensor(const aclTensor* tensor, int64_t size)
{
    return tensor == nullptr || (IsInt32Tensor(tensor) && tensor->GetViewShape().GetDimNum() == 1 &&
                                 tensor->GetViewShape().GetDim(0) == size);
}
} // namespace

extern "C" {
aclnnStatus aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional, int64_t numHeadsQ, int64_t numHeadsK,
    int64_t headDim, int64_t topk, int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t candidateTopkBlocks,
    int64_t candidateBlockSize, const aclTensor* metadata, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (layoutQOptional == nullptr || layoutKOptional == nullptr || metadata == nullptr || workspaceSize == nullptr ||
        executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required input, output or executor argument is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if ((numHeadsQ != QUERY_HEAD_NUM_32 && numHeadsQ != QUERY_HEAD_NUM_64) || numHeadsK != KEY_HEAD_NUM ||
        headDim != HEAD_DIM || topk < 1 || topk > TOPK_MAX || quantMode != QUANT_MODE_INT8 || batchSize < 0 ||
        batchSize == std::numeric_limits<int64_t>::max() || maxSeqlenQ < -1 || maxSeqlenK < -1 ||
        (maskMode != MASK_MODE_NONE && maskMode != MASK_MODE_CAUSAL) || cmpRatio < 1 || cmpRatio > CMP_RATIO_MAX ||
        (cmpRatio & (cmpRatio - 1)) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid metadata scalar parameters.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const std::string layoutQ = layoutQOptional;
    const std::string layoutK = layoutKOptional;
    const bool isTnd = layoutQ == "TND";
    if ((!isTnd && layoutQ != "BSND") || layoutK != "PA_BBND" || cuSeqlensKOptional != nullptr ||
        (!isTnd && cuSeqlensQOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid layouts or cumulative sequence inputs.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((isTnd && cuSeqlensQOptional == nullptr) || sequsedKOptional == nullptr ||
        (maskMode == MASK_MODE_CAUSAL && cmpRatio != 1 && cmpResidualKOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required sequence input is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!IsSequenceTensor(cuSeqlensQOptional, batchSize + 1) || !IsSequenceTensor(sequsedQOptional, batchSize) ||
        !IsSequenceTensor(sequsedKOptional, batchSize) || !IsSequenceTensor(cmpResidualKOptional, batchSize) ||
        !IsInt32Tensor(metadata) || metadata->GetViewShape().GetDimNum() != 1 ||
        metadata->GetViewShape().GetDim(0) != METADATA_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid sequence tensor or metadata output descriptor.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((candidateTopkBlocks == -1 && candidateBlockSize == -1) ||
          (candidateTopkBlocks == CANDIDATE_TOPK_BLOCKS && candidateBlockSize == CANDIDATE_BLOCK_SIZE))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Candidate attributes must be (-1, -1) or (2048, 8).");
        return ACLNN_ERR_PARAM_INVALID;
    }

    L2_DFX_PHASE_1(
        aclnnQuantLightningIndexerV3Metadata,
        DFX_IN(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional,
               numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional,
               layoutKOptional, maskMode, cmpRatio, candidateTopkBlocks, candidateBlockSize),
        DFX_OUT(metadata));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const auto& npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &aicCoreNum) != ACL_SUCCESS) {
        aicCoreNum = npuInfo.GetCubeCoreNum();
    }
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreNum) != ACL_SUCCESS) {
        aivCoreNum = npuInfo.GetVectorCoreNum();
    }
    const std::string socVersion = npuInfo.GetSocLongVersion();
    auto output = l0op::QuantLightningIndexerV3Metadata(
        cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional, numHeadsQ,
        numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional,
        maskMode, cmpRatio, candidateTopkBlocks, candidateBlockSize, aicCoreNum, aivCoreNum, socVersion.c_str(),
        metadata, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantLightningIndexerV3Metadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantLightningIndexerV3Metadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
