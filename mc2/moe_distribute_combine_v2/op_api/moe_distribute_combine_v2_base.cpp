/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_distribute_combine_v2_base.cpp
 * \brief
 */
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "opdev/common_types.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "moe_distribute_combine_v2_base.h"
#include "aclnnInner_moe_distribute_combine_v2.h"

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM (HCCL_CHANNEL_SUPPORT_VERSION)
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#endif

using namespace Ops::Transformer;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
extern aclnnStatus aclnnInnerMoeDistributeCombineV3GetWorkspaceSize(
    const aclTensor *context, const aclTensor *expandX, const aclTensor *expertIds,
    const aclTensor *assistInfoForCombine, const aclTensor *epSendCounts, const aclTensor *expertScales,
    const aclTensor *tpSendCounts, const aclTensor *xActiveMask, const aclTensor *activationScale,
    const aclTensor *weightScale, const aclTensor *groupList, const aclTensor *expandScales,
    const aclTensor *sharedExpertX, const aclTensor *elasticInfo, const aclTensor *oriX,
    const aclTensor *constExpertAlpha1, const aclTensor *constExpertAlpha2, const aclTensor *constExpertV,
    const aclTensor *performanceInfo, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t cclBufferSize, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    const char *commAlg, int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor *x,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeDistributeCombineV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream);

enum class CommType : uint64_t {
    AIV = 0, // AIV通信设置为0
    CCU = 1  // ccu通信设置为1
};
#endif

bool CombineCheckNotNull(const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
                         const aclTensor *epSendCounts, const aclTensor *expertScales, const char *groupEp,
                         aclTensor *x)
{
    OP_CHECK_NULL(epSendCounts, return false);
    OP_CHECK_NULL(expertScales, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);

    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMoeDistributeCombine", "groupEp");
        return false;
    }
    return true;
}

aclnnStatus CombineCheckParams(const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *expandIdx,
                               const aclTensor *epSendCounts, const aclTensor *expertScales, const char *groupEp,
                               const char *groupTp, aclTensor *x)
{
    CHECK_RET(CombineCheckNotNull(expandX, expertIds, expandIdx, epSendCounts, expertScales, groupEp, x),
              ACLNN_ERR_PARAM_NULLPTR);
    // groupTp为预留参数(当前版本不支持TP域通信)，nullptr视同空串，避免后续strnlen空指针解引用
    if (groupTp == nullptr) {
        groupTp = "";
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
    return ACLNN_SUCCESS;
}

static inline void SafeCopyGroupBuf(char *dst, size_t dstSize, const char *src, size_t maxCopyLen)
{
    if (src != nullptr) {
        (void)strncpy_s(dst, dstSize, src, maxCopyLen);
    }
}

static void SetCommArgs(aclOpExecutor **executor, const bool is910B, const bool is950, const char *commAlg)
{
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    if (is950) {
        CommType type = CommType::AIV;
        if (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            type = CommType::CCU;
        }
        void *args = reinterpret_cast<void *>(static_cast<uint64_t>(type));
        NnopbaseSetUserHandle(*executor, args);
    }
#endif

    if (NnopbaseSetHcclServerType) {
        if (is910B) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else if (is950 && commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
}

aclnnStatus aclnnMoeDistributeCombineBaseGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *tpSendCountsOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *activationScaleOptional,
    const aclTensor *weightScaleOptional, const aclTensor *groupListOptional, const aclTensor *expandScalesOptional,
    const aclTensor *sharedExpertXOptional, const aclTensor *elasticInfoOptional, const aclTensor *oriXOptional,
    const aclTensor *constExpertAlpha1Optional, const aclTensor *constExpertAlpha2Optional,
    const aclTensor *constExpertVOptional, const aclTensor *performanceInfoOptional, const char *groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char *groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
    int64_t outDtype, int64_t commQuantMode, int64_t groupListType, const char *commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum, aclTensor *xOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("enter to the  aclnnMoeDistributeCombineBaseGetWorkspaceSize\n");
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    auto retParam = CombineCheckParams(expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, groupEp,
                                       groupTp, xOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclTensor *mc2Context = nullptr;
    char groupEpBuf[HCCL_GROUP_NAME_MAX] = {0};
    SafeCopyGroupBuf(groupEpBuf, HCCL_GROUP_NAME_MAX, groupEp, HCCL_GROUP_NAME_MAX - 1);
    char groupTpBuf[HCCL_GROUP_NAME_MAX] = {0};
    const char *groupTpCombineV2Temp = groupTp;
    if (is910B) {
        groupTpCombineV2Temp = "";
    }
    SafeCopyGroupBuf(groupTpBuf, HCCL_GROUP_NAME_MAX, groupTpCombineV2Temp, HCCL_GROUP_NAME_MAX - 1);
    char commAlgBuf[HCCL_GROUP_NAME_MAX] = {0};
    SafeCopyGroupBuf(commAlgBuf, HCCL_GROUP_NAME_MAX, commAlg, HCCL_GROUP_NAME_MAX - 1);

    // ccu暂时不支持performanceInfo
    if (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
        if (performanceInfoOptional != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("MoeDistributeCombineV2", "performanceInfo", "not nullptr",
                                                  "performanceInfo not supported in ccu");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }

    aclnnStatus getWorkspaceSizesRes = ACLNN_ERR_INNER;
    if (!is950 || (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0)) { // ccu暂时不支持mc2Context
        getWorkspaceSizesRes = aclnnInnerMoeDistributeCombineV2GetWorkspaceSize(
            expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, tpSendCountsOptional,
            xActiveMaskOptional, activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional,
            sharedExpertXOptional, elasticInfoOptional, oriXOptional, constExpertAlpha1Optional,
            constExpertAlpha2Optional, constExpertVOptional, performanceInfoOptional, groupEpBuf, epWorldSize, epRankId,
            moeExpertNum, groupTpBuf, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum,
            globalBs, outDtype, commQuantMode, groupListType, commAlgBuf, zeroExpertNum, copyExpertNum, constExpertNum,
            xOut, workspaceSize, executor);
    } else {
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
        uint64_t hcclBuffSize = 0;
        const char *opName = "moe_distribute_v2";
        auto aclnnRet = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(groupEp, opName, hcclBuffSize, mc2Context);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        getWorkspaceSizesRes = aclnnInnerMoeDistributeCombineV3GetWorkspaceSize(
            mc2Context, expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, tpSendCountsOptional,
            xActiveMaskOptional, activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional,
            sharedExpertXOptional, elasticInfoOptional, oriXOptional, constExpertAlpha1Optional,
            constExpertAlpha2Optional, constExpertVOptional, performanceInfoOptional, epWorldSize, epRankId,
            moeExpertNum, hcclBuffSize, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum,
            globalBs, outDtype, commQuantMode, groupListType, commAlgBuf, zeroExpertNum, copyExpertNum, constExpertNum,
            xOut, workspaceSize, executor);
#endif
    }
    SetCommArgs(executor, is910B, is950, commAlg);
    return getWorkspaceSizesRes;
}

// aclnn二段式接口
aclnnStatus aclnnMoeDistributeCombineBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    if (is950) {
        void *arg = NnopbaseGetUserHandle(executor);
        uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
        CommType commType = static_cast<CommType>(handleVal);
        if (commType == CommType::AIV) {
            OP_LOGD("aclnn_combine inner v3 start");
            return aclnnInnerMoeDistributeCombineV3(workspace, workspaceSize, executor, stream);
        }
    }
#endif
    OP_LOGD("aclnn_combine inner v2 start");
    return aclnnInnerMoeDistributeCombineV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
