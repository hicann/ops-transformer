/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_lightning_indexer_v2.cpp
 * \brief lightning_indexer_v2 手写 aclnn 兼容接口：签名冻结为已发布的 V2 原型
 *（无 candidate 参数），candidate 能力由 aclnnLightningIndexerV3 承接。
 */

#include <string.h>
#include <vector>
#include "graph/types.h"
#include "aclnn_lightning_indexer_v2.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerLightningIndexerV2GetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* w, const aclTensor* cuSeqlensQOptional,
    const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKOptional,
    const aclTensor* cmpResidualKOptional, const aclTensor* blockTableOptional,
    const aclTensor* outputIdxOffsetOptional, const aclTensor* metadataOptional, int64_t topk, int64_t maxSeqlenQ,
    char* layoutQ, char* layoutK, int64_t maskMode, int64_t cmpRatio, int64_t returnValue, int64_t candidateTopkBlocks,
    int64_t candidateBlockSize, const aclTensor* sparseIndices, const aclTensor* sparseValues,
    const aclTensor* candidateBlockIndicesOutOptional, const aclTensor* candidateBlockLengthOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerLightningIndexerV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                const aclrtStream stream);

// candidate 关闭哨兵与缺省块大小，与 op_host/lightning_indexer_v2_tiling.h 中
// CANDIDATE_TOPK_BLOCKS_OFF 及 def 缺省值保持一致
constexpr int64_t CANDIDATE_TOPK_BLOCKS_OFF = -1;
constexpr int64_t CANDIDATE_BLOCK_SIZE_DEFAULT = 8;

// V2 原型无 candidate 出参，转发 inner 前以 (0,) int32 空 tensor 占位，
// 与 candidate 关闭时 infershape 给出的 candidate 输出形状一致
class LightningIndexerV2CandidateOutputHolder {
public:
    LightningIndexerV2CandidateOutputHolder()
    {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        candidateBlockIndices_ = aclCreateTensor(shape.data(), shape.size(), ACL_INT32, shape.data(), 0, ACL_FORMAT_ND,
                                                 shape.data(), shape.size(), static_cast<void*>(&addr));
        candidateBlockLength_ = aclCreateTensor(shape.data(), shape.size(), ACL_INT32, shape.data(), 0, ACL_FORMAT_ND,
                                                shape.data(), shape.size(), static_cast<void*>(&addr));
    }

    ~LightningIndexerV2CandidateOutputHolder()
    {
        if (candidateBlockIndices_ != nullptr) {
            aclDestroyTensor(candidateBlockIndices_);
            candidateBlockIndices_ = nullptr;
        }
        if (candidateBlockLength_ != nullptr) {
            aclDestroyTensor(candidateBlockLength_);
            candidateBlockLength_ = nullptr;
        }
    }

    const aclTensor* CandidateTopkIndices() const
    {
        return candidateBlockIndices_;
    }

    const aclTensor* CandidateBlockLength() const
    {
        return candidateBlockLength_;
    }

private:
    const aclTensor* candidateBlockIndices_ = nullptr;
    const aclTensor* candidateBlockLength_ = nullptr;
};

aclnnStatus aclnnLightningIndexerV2GetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* w, const aclTensor* cuSeqlensQOptional,
    const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKOptional,
    const aclTensor* cmpResidualKOptional, const aclTensor* blockTableOptional,
    const aclTensor* outputIdxOffsetOptional, const aclTensor* metadataOptional, int64_t topk, int64_t maxSeqlenQ,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t returnValue,
    const aclTensor* sparseIndices, const aclTensor* sparseValues, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (q == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "q pointer is null, cannot get data type!");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (k == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "k pointer is null, cannot get data type!");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (w == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "w pointer is null, cannot get data type!");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (sparseIndices == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sparseIndices cannot be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (sparseValues == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sparseValues cannot be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize or executor cannot be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    LightningIndexerV2CandidateOutputHolder candidateOutputHolder;
    if (candidateOutputHolder.CandidateTopkIndices() == nullptr ||
        candidateOutputHolder.CandidateBlockLength() == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the candidate output placeholder of LightningIndexerV2!");
        return ge::GRAPH_FAILED;
    }

    // V2 兼容接口固定 candidate 关闭：转发语义与已发布版本（单级 TopK）一致
    return aclnnInnerLightningIndexerV2GetWorkspaceSize(
        q, k, w, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional,
        blockTableOptional, outputIdxOffsetOptional, metadataOptional, topk, maxSeqlenQ, layoutQOptional,
        layoutKOptional, maskMode, cmpRatio, returnValue, CANDIDATE_TOPK_BLOCKS_OFF, CANDIDATE_BLOCK_SIZE_DEFAULT,
        sparseIndices, sparseValues, candidateOutputHolder.CandidateTopkIndices(),
        candidateOutputHolder.CandidateBlockLength(), workspaceSize, executor);
}

aclnnStatus aclnnLightningIndexerV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream)
{
    return aclnnInnerLightningIndexerV2(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
