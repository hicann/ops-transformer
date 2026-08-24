/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnnInner_moe_ep_combine_epilogue.h"

using namespace op;

namespace {
constexpr int64_t NETWORK_DIRECT = 0;
constexpr int64_t NETWORK_HYBRID = 1;
} // namespace

static aclnnStatus CheckNotNull(const aclTensor *context, const aclTensor *topkIdx, aclTensor *combinedX)
{
    CHECK_RET(context != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(topkIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(combinedX != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(int64_t epWorldSize, int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank,
                               int64_t cclBufferSize, int64_t topoType, int64_t rankNumPerServer)
{
    CHECK_RET(epWorldSize > 1, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(epRankId >= 0 && epRankId < epWorldSize, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numExperts % epWorldSize == 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numMaxTokensPerRank > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(cclBufferSize > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(topoType == NETWORK_DIRECT || topoType == NETWORK_HYBRID, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(rankNumPerServer > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(epWorldSize % rankNumPerServer == 0, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE
};

aclnnStatus MoeEpCombineEpilogueGetWorkspaceSize(const aclTensor *context, const aclTensor *topkIdx,
                                                 int64_t epWorldSize, int64_t epRankId, int64_t numExperts,
                                                 int64_t numMaxTokensPerRank, int64_t cclBufferSize,
                                                 bool hasTopkWeights, int64_t topoType, int64_t rankNumPerServer,
                                                 aclTensor *combinedX, aclTensor *combinedTopkWeights,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("MoeEpCombineEpilogue", "Begin to do MoeEpCombineEpilogueGetWorkspaceSize");
    auto retNotNull = CheckNotNull(context, topkIdx, combinedX);
    CHECK_RET(retNotNull == ACLNN_SUCCESS, retNotNull);
    auto retParams =
        CheckParams(epWorldSize, epRankId, numExperts, numMaxTokensPerRank, cclBufferSize, topoType, rankNumPerServer);
    CHECK_RET(retParams == ACLNN_SUCCESS, retParams);

    return aclnnInnerMoeEpCombineEpilogueGetWorkspaceSize(
        context, topkIdx, epWorldSize, epRankId, numExperts, numMaxTokensPerRank, cclBufferSize, hasTopkWeights,
        topoType, rankNumPerServer, combinedX, combinedTopkWeights, workspaceSize, executor);
}

aclnnStatus aclnnMoeEpCombineEpilogueGetWorkspaceSize(const aclTensor *context, const aclTensor *topkIdx,
                                                      int64_t epWorldSize, int64_t epRankId, int64_t numExperts,
                                                      int64_t numMaxTokensPerRank, int64_t cclBufferSize,
                                                      bool hasTopkWeights, int64_t topoType, int64_t rankNumPerServer,
                                                      aclTensor *combinedX, aclTensor *combinedTopkWeights,
                                                      uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return MoeEpCombineEpilogueGetWorkspaceSize(
        context, topkIdx, epWorldSize, epRankId, numExperts, numMaxTokensPerRank, cclBufferSize, hasTopkWeights,
        topoType, rankNumPerServer, combinedX, combinedTopkWeights, workspaceSize, executor);
}

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeEpCombineEpilogue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerMoeEpCombineEpilogue(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
