/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_distribute_dispatch_v3.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeDistributeDispatchV5GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *expertScalesOptional, const aclTensor *elasticInfoOptional,
    const aclTensor *performanceInfoOptional, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t cclBufferSize, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char *commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor *expandXOut,
    aclTensor *dynamicScalesOut, aclTensor *assistInfoForCombineOut, aclTensor *expertTokenNumsOut,
    aclTensor *epRecvCountsOut, aclTensor *tpRecvCountsOut, aclTensor *expandScalesOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_LOGD("aclnn_dispatch v5 WorkspaceSize start");
    int64_t yDtype = expandXOut->GetDataType();
    aclnnStatus getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV3GetWorkspaceSize(
        context, x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, elasticInfoOptional,
        performanceInfoOptional, epWorldSize, epRankId, moeExpertNum, cclBufferSize, tpWorldSize, tpRankId,
        expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType,
        const_cast<char *>(commAlg), zeroExpertNum, copyExpertNum, constExpertNum, yDtype, expandXOut, dynamicScalesOut,
        assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize,
        executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDistributeDispatchV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    OP_LOGD("aclnn_dispatch v5 start");
    return aclnnInnerMoeDistributeDispatchV3(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
