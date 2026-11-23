/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_lightning_indexer_v3.h"
#include "opdev/op_log.h"

extern "C" {
aclnnStatus aclnnInnerQuantLightningIndexerV2GetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* w, const aclTensor* qDescale, const aclTensor* kDescale,
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional, const aclTensor* blockTableOptional,
    const aclTensor* outputIdxOffsetOptional, const aclTensor* metadataOptional,
    const aclTensor* causalSeqlensKvGlobalOptional, int64_t topk, int64_t quantMode, int64_t maxSeqlenQ,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t returnValue,
    int64_t candidateTopkBlocks, int64_t candidateBlockSize, const aclIntArray* dcpParams,
    const aclTensor* sparseIndicesOut, const aclTensor* sparseValuesOut, const aclTensor* candidateBlockIndicesOut,
    const aclTensor* candidateBlockLengthOut, uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnInnerQuantLightningIndexerV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream);
}

namespace {
constexpr int64_t CANDIDATE_OFF = -1;
constexpr int64_t CANDIDATE_TOPK_BLOCKS = 2048;
constexpr int64_t CANDIDATE_BLOCK_SIZE = 8;

class CandidateOutputHolder {
public:
    explicit CandidateOutputHolder(const aclTensor*& output)
    {
        if (output == nullptr) {
            const int64_t shape[] = {0};
            const int64_t strides[] = {1};
            tensor_ = aclCreateTensor(shape, 1, ACL_INT32, strides, 0, ACL_FORMAT_ND, shape, 1, &storage_);
            output = tensor_;
        }
    }

    ~CandidateOutputHolder()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
    }

    CandidateOutputHolder(const CandidateOutputHolder&) = delete;
    CandidateOutputHolder& operator=(const CandidateOutputHolder&) = delete;

private:
    int32_t storage_ = 0;
    const aclTensor* tensor_ = nullptr;
};
} // namespace

extern "C" {
aclnnStatus aclnnQuantLightningIndexerV3GetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* w, const aclTensor* qDescale, const aclTensor* kDescale,
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional, const aclTensor* blockTableOptional,
    const aclTensor* outputIdxOffsetOptional, const aclTensor* metadataOptional,
    const aclTensor* causalSeqlensKvGlobalOptional, int64_t topk, int64_t quantMode, int64_t maxSeqlenQ,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t returnValue,
    int64_t candidateTopkBlocks, int64_t candidateBlockSize, const aclIntArray* dcpParams,
    const aclTensor* sparseIndicesOut, const aclTensor* sparseValuesOut, const aclTensor* candidateBlockIndicesOut,
    const aclTensor* candidateBlockLengthOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (q == nullptr || k == nullptr || w == nullptr || qDescale == nullptr || kDescale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required input, output or executor argument is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (sparseIndicesOut == nullptr || sparseValuesOut == nullptr || workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required input, output or executor argument is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const bool candidateOff = candidateTopkBlocks == CANDIDATE_OFF && candidateBlockSize == CANDIDATE_OFF;
    const bool candidateOn = candidateTopkBlocks == CANDIDATE_TOPK_BLOCKS && candidateBlockSize == CANDIDATE_BLOCK_SIZE;
    if (!candidateOff && !candidateOn) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Candidate attributes must be (-1, -1) or (2048, 8).");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (candidateOn && (candidateBlockIndicesOut == nullptr || candidateBlockLengthOut == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Both candidate outputs are required when candidate mode is enabled.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (dcpParams != nullptr) {
        uint64_t dcpParamsSize = 0U;
        aclGetIntArraySize(dcpParams, &dcpParamsSize);
        if (dcpParamsSize != 3U) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "dcpParams must contain exactly 3 elements: [cpWorld, cpRank, cpInterleaveSize].");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // dcpParams 的具体取值（cpWorld/cpRank/cpInterleaveSize）及 DCP 组合约束由算子 tiling 统一校验。

    CandidateOutputHolder indicesHolder(candidateBlockIndicesOut);
    CandidateOutputHolder lengthHolder(candidateBlockLengthOut);
    if (candidateBlockIndicesOut == nullptr || candidateBlockLengthOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to create an empty candidate output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    aclnnStatus ret = aclnnInnerQuantLightningIndexerV2GetWorkspaceSize(
        q, k, w, qDescale, kDescale, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
        cmpResidualKOptional, blockTableOptional, outputIdxOffsetOptional, metadataOptional,
        causalSeqlensKvGlobalOptional, topk, quantMode, maxSeqlenQ, layoutQOptional, layoutKOptional, maskMode,
        cmpRatio, returnValue, candidateTopkBlocks, candidateBlockSize, dcpParams, sparseIndicesOut, sparseValuesOut,
        candidateBlockIndicesOut, candidateBlockLengthOut, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnQuantLightningIndexerV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    return aclnnInnerQuantLightningIndexerV2(workspace, workspaceSize, executor, stream);
}
}
