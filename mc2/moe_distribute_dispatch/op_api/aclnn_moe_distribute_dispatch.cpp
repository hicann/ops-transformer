/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_dispatch.h"
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "opdev/common_types.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "aclnnInner_moe_distribute_dispatch.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t DISPATCH_DYNAMIC_QUANT_MODE = 2;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor *x, const aclTensor *expertIds, const char *groupEp, const aclTensor *expandX,
                         const aclTensor *expandIdx, const aclTensor *expertTokensNums, const aclTensor *epRecvCounts)
{
    OP_LOGD("aclnn_moe_distribute_dispatch CheckNotNull start");
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expandIdx, return false);
    OP_CHECK_NULL(expertTokensNums, return false);
    OP_CHECK_NULL(epRecvCounts, return false);
    OP_LOGD("aclnn_moe_distribute_dispatch CheckNotNull success");
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMoeDistributeDispatch", "groupEp");
        return false;
    }
    return true;
}

static inline bool SafeCopyGroupBuf(char *dst, size_t dstSize, const char *src, size_t maxCopyLen)
{
    if (src != nullptr) {
        error_t ret = strncpy_s(dst, dstSize, src, maxCopyLen);
        if (ret != EOK) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "aclnnMoeDistributeDispatch", "group strncpy_s return value",
                std::to_string(static_cast<int32_t>(ret)).c_str(), "strncpy_s failed");
            return false;
        }
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *expertIds, const char *groupEp, const char *groupTp,
                               int64_t quantMode, const aclTensor *expandX, const aclTensor *dynamicScales,
                               const aclTensor *expandIdx, const aclTensor *expertTokensNums,
                               const aclTensor *epRecvCounts)
{
    OP_LOGD("aclnn_moe_distribute_dispatch CheckParams start");
    CHECK_RET(CheckNotNull(x, expertIds, groupEp, expandX, expandIdx, expertTokensNums, epRecvCounts),
              ACLNN_ERR_PARAM_NULLPTR);
    // groupTp为预留参数(当前版本不支持TP域通信)，nullptr视同空串，避免后续strnlen空指针解引用
    if (groupTp == nullptr) {
        groupTp = "";
    }
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    if (is910B) {
        OP_LOGD("A2 platform, groupTp should be empty");
        CHECK_RET(strnlen(groupTp, HCCL_GROUP_NAME_MAX) == 0, ACL_ERROR_INVALID_PARAM);
    }
    if (quantMode == DISPATCH_DYNAMIC_QUANT_MODE) {
        OP_LOGD("quantMode = 2, dynamicScales can't be null");
        CHECK_RET(dynamicScales != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnMoeDistributeDispatch", "groupEp", std::to_string(strnlen(groupEp, HCCL_GROUP_NAME_MAX)).c_str(),
            std::string("groupEp name exceeds ").append(std::to_string(HCCL_GROUP_NAME_MAX)).c_str());
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnMoeDistributeDispatch", "groupTp", std::to_string(strnlen(groupTp, HCCL_GROUP_NAME_MAX)).c_str(),
            std::string("groupTp name exceeds ").append(std::to_string(HCCL_GROUP_NAME_MAX)).c_str());
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    OP_LOGD("aclnn_moe_distribute_dispatch CheckParams success");
    return ACLNN_SUCCESS;
}

aclnnStatus MoeDistributeDispatchGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales, const aclTensor *xActiveMask,
    const aclTensor *expertScales, const char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t shareExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, aclTensor *expandX,
    aclTensor *dynamicScales, aclTensor *expandIdx, aclTensor *expertTokensNums, aclTensor *epRecvCounts,
    aclTensor *tpRecvCounts, aclTensor *expandScales, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnMoeDistributeDispatchGetWorkspaceSize start");
    auto ret_param = CheckParams(x, expertIds, groupEp, groupTp, quantMode, expandX, dynamicScales, expandIdx,
                                 expertTokensNums, epRecvCounts);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    char groupEpBuf[HCCL_GROUP_NAME_MAX] = {0};
    CHECK_RET(SafeCopyGroupBuf(groupEpBuf, HCCL_GROUP_NAME_MAX, groupEp, HCCL_GROUP_NAME_MAX - 1), ACLNN_ERR_INNER);
    char groupTpBuf[HCCL_GROUP_NAME_MAX] = {0};
    CHECK_RET(SafeCopyGroupBuf(groupTpBuf, HCCL_GROUP_NAME_MAX, groupTp, HCCL_GROUP_NAME_MAX - 1), ACLNN_ERR_INNER);

    aclnnStatus ret = aclnnInnerMoeDistributeDispatchGetWorkspaceSize(
        x, expertIds, scales, xActiveMask, expertScales, groupEpBuf, epWorldSize, epRankId,
        moeExpertNum, groupTpBuf, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
        shareExpertRankNum, quantMode, globalBs, expertTokenNumsType, expandX, dynamicScales, expandIdx,
        expertTokensNums, epRecvCounts, tpRecvCounts, expandScales, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales, const aclTensor *xActiveMask,
    const aclTensor *expertScales, const char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, aclTensor *expandX,
    aclTensor *dynamicScales, aclTensor *expandIdx, aclTensor *expertTokenNums, aclTensor *epRecvCounts,
    aclTensor *tpRecvCounts, aclTensor *expandScales, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const char *groupTpEmptyWord = "";
        aclnnStatus distributeDispatchRet = MoeDistributeDispatchGetWorkspaceSize(
            x, expertIds, scales, xActiveMask, expertScales, groupEp, epWorldSize, epRankId, moeExpertNum,
            groupTpEmptyWord, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode,
            globalBs, expertTokenNumsType, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
            tpRecvCounts, expandScales, workspaceSize, executor);
        return distributeDispatchRet;
    }
    aclnnStatus ret = MoeDistributeDispatchGetWorkspaceSize(
        x, expertIds, scales, xActiveMask, expertScales, groupEp, epWorldSize, epRankId, moeExpertNum, groupTp,
        tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs,
        expertTokenNumsType, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts, tpRecvCounts,
        expandScales, workspaceSize, executor);
    return ret;
}
aclnnStatus aclnnMoeDistributeDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    aclnnStatus ret = aclnnInnerMoeDistributeDispatch(workspace, workspaceSize, executor, stream);
    return ret;
}
#ifdef __cplusplus
}
#endif
