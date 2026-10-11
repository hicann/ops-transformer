/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_combine.h"
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "opdev/common_types.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "aclnnInner_moe_distribute_combine.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx,
                         const aclTensor* epSendCounts, const aclTensor* expertScales, const char* groupEp,
                         aclTensor* x)
{
    OP_LOGD("aclnn_moe_distribute_combine CheckNotNull start");
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(expandIdx, return false);
    OP_CHECK_NULL(epSendCounts, return false);
    OP_CHECK_NULL(expertScales, return false);
    OP_CHECK_NULL(x, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMoeDistributeCombine", "groupEp");
        return false;
    }
    OP_LOGD("aclnn_moe_distribute_combine CheckNotNull success");
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx,
                               const aclTensor* epSendCounts, const aclTensor* expertScales, const char* groupEp,
                               const char* groupTp, aclTensor* x)
{
    OP_LOGD("aclnn_moe_distribute_combine checkparams start");
    CHECK_RET(CheckNotNull(expandX, expertIds, expandIdx, epSendCounts, expertScales, groupEp, x),
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
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnMoeDistributeCombine", "groupEp", std::to_string(strnlen(groupEp, HCCL_GROUP_NAME_MAX)).c_str(),
            std::string("groupEp name exceeds ").append(std::to_string(HCCL_GROUP_NAME_MAX)).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnMoeDistributeCombine", "groupTp", std::to_string(strnlen(groupTp, HCCL_GROUP_NAME_MAX)).c_str(),
            std::string("groupTp name exceeds ").append(std::to_string(HCCL_GROUP_NAME_MAX)).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("aclnn_moe_distribute_combine checkparams success");
    return ACLNN_SUCCESS;
}

aclnnStatus MoeDistributeCombineGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* tpSendCounts, const aclTensor* xActiveMask,
    const aclTensor* activationScale, const aclTensor* weightScale, const aclTensor* groupList,
    const aclTensor* expandScales, const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    aclTensor* x, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret_param = CheckParams(expandX, expertIds, expandIdx, epSendCounts, expertScales, groupEp, groupTp, x);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = aclnnInnerMoeDistributeCombineGetWorkspaceSize(
        expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
        weightScale, groupList, expandScales, const_cast<char*>(groupEp), epWorldSize, epRankId, moeExpertNum,
        const_cast<char*>(groupTp), tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum,
        globalBs, outDtype, commQuantMode, groupListType, x, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMoeDistributeCombineGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* tpSendCounts, const aclTensor* xActiveMask,
    const aclTensor* activationScale, const aclTensor* weightScale, const aclTensor* groupList,
    const aclTensor* expandScales, const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    aclTensor* x, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const char* groupTpEmptyWord = "";
        aclnnStatus distributeCombineRet = MoeDistributeCombineGetWorkspaceSize(
            expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
            weightScale, groupList, expandScales, groupEp, epWorldSize, epRankId, moeExpertNum, groupTpEmptyWord,
            tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
            commQuantMode, groupListType, x, workspaceSize, executor);
        return distributeCombineRet;
    }
    aclnnStatus ret = MoeDistributeCombineGetWorkspaceSize(
        expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
        weightScale, groupList, expandScales, groupEp, epWorldSize, epRankId, moeExpertNum, groupTp, tpWorldSize,
        tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode,
        groupListType, x, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMoeDistributeCombine(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    aclnnStatus ret = aclnnInnerMoeDistributeCombine(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
