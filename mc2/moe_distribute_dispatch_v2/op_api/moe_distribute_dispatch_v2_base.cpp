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
 * \file moe_distribute_dispatch_v2_base.cpp
 * \brief
 */

#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "mc2_log_compat.h"
#include "moe_distribute_dispatch_v2_base.h"
#include "aclnnInner_moe_distribute_dispatch_v2.h"

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
extern aclnnStatus aclnnInnerMoeDistributeDispatchV3GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales,
    const aclTensor *xActiveMask, const aclTensor *expertScales, const aclTensor *elasticInfo,
    const aclTensor *performanceInfo, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t cclBufferSize, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t shareExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char *commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, int64_t ydtype, aclTensor *expandX,
    aclTensor *dynamicScales, aclTensor *assist_info_for_combine, aclTensor *expertTokensNums, aclTensor *epRecvCounts,
    aclTensor *tpRecvCounts, aclTensor *expandScales, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeDistributeDispatchV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream);

enum class CommType : uint64_t {
    AIV = 0, // AIV通信设置为0
    CCU = 1  // ccu通信设置为1
};
#endif

bool DispatchCheckNotNull(const aclTensor *x, const aclTensor *expertIds, const char *groupEp,
                          [[maybe_unused]] const char *groupTp, aclTensor *expandX,
                          [[maybe_unused]] aclTensor *dynamicScales, aclTensor *assistInfoForCombine,
                          aclTensor *expertTokensNums, aclTensor *epRecvCounts)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(expertTokensNums, return false);
    OP_CHECK_NULL(epRecvCounts, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMoeDistributeDispatchV2", "groupEp");
        return false;
    }
    return true;
}

aclnnStatus DispatchCheckParams(const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
                                const char *groupEp, const char *groupTp, int64_t quantMode, aclTensor *expandX,
                                aclTensor *dynamicScales, aclTensor *assistInfoForCombine, aclTensor *expertTokensNums,
                                aclTensor *epRecvCounts, aclTensor *tpRecvCounts)
{
    CHECK_RET(DispatchCheckNotNull(x, expertIds, groupEp, groupTp, expandX, dynamicScales, assistInfoForCombine,
                                   expertTokensNums, epRecvCounts),
              ACLNN_ERR_PARAM_NULLPTR);
    // groupTp为预留参数(当前版本不支持TP域通信)，nullptr视同空串，避免后续strnlen空指针解引用
    if (groupTp == nullptr) {
        groupTp = "";
    }

    if (quantMode == DISPATCH_DYNAMIC_QUANT_MODE) {
        OP_LOGD("quantMode = 2, dynamicScales can't be null");
        CHECK_RET(dynamicScales != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (x->GetDataType() == ACL_INT32) {
        OP_LOGD("x dtype is INT32, scales and dynamicScales can't be null");
        CHECK_RET(scalesOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(dynamicScales != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMoeDistributeDispatchV2", "groupEp",
                                              "length exceeds " + std::to_string(HCCL_GROUP_NAME_MAX),
                                              "groupEp name too long");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMoeDistributeDispatchV2", "groupTp",
                                              "length exceeds " + std::to_string(HCCL_GROUP_NAME_MAX),
                                              "groupTp name too long");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static inline void SafeCopyGroupBuf(char *dst, size_t dstSize, const char *src, size_t maxCopyLen)
{
    if (src != nullptr) {
        (void)strncpy_s(dst, dstSize, src, maxCopyLen);
    }
}

static void SetCommArgs(const bool is950, const bool is910B, const char *commAlg, aclOpExecutor **executor)
{
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    if (is950) {
        void *arg = reinterpret_cast<void *>(static_cast<uintptr_t>(CommType::AIV)); // 默认MTE为0
        if (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            arg = reinterpret_cast<void *>(static_cast<uintptr_t>(CommType::CCU)); // ccu为1
        }
        NnopbaseSetUserHandle(*executor, arg);
    }
#endif

    if (NnopbaseSetHcclServerType) { // 给ACLnn框架指定通讯方式。
        if (is910B) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else if (is950 && commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
}

aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSizeBase(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *expertScalesOptional, const aclTensor *elasticInfoOptional,
    const aclTensor *performanceInfoOptional, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, const char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType,
    int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
    int64_t expertTokenNumsType, const char *commAlg, int64_t zeroExpertNum, int64_t copyExpertNum,
    int64_t constExpertNum, aclTensor *expandXOut, aclTensor *dynamicScalesOut, aclTensor *assistInfoForCombineOut,
    aclTensor *expertTokenNumsOut, aclTensor *epRecvCountsOut, aclTensor *tpRecvCountsOut, aclTensor *expandScalesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("enter to the  aclnnMoeDistributeDispatchGetWorkspaceSizeBase\n");
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    auto retParam =
        DispatchCheckParams(x, expertIds, scalesOptional, groupEp, groupTp, quantMode, expandXOut, dynamicScalesOut,
                            assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    aclTensor *mc2Context = nullptr;
    aclnnStatus getWorkspaceSizesRes = ACLNN_ERR_INNER;
    const char *groupTpDispatchV2Temp = groupTp;
    if (is910B) {
        groupTpDispatchV2Temp = "";
    }

    char groupEpBuf[HCCL_GROUP_NAME_MAX] = {0};
    SafeCopyGroupBuf(groupEpBuf, HCCL_GROUP_NAME_MAX, groupEp, HCCL_GROUP_NAME_MAX - 1);
    char groupTpBuf[HCCL_GROUP_NAME_MAX] = {0};
    SafeCopyGroupBuf(groupTpBuf, HCCL_GROUP_NAME_MAX, groupTpDispatchV2Temp, HCCL_GROUP_NAME_MAX - 1);
    char commAlgBuf[HCCL_GROUP_NAME_MAX] = {0};
    SafeCopyGroupBuf(commAlgBuf, HCCL_GROUP_NAME_MAX, commAlg, HCCL_GROUP_NAME_MAX - 1);

    // ccu暂时不支持performanceInfo
    if (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
        if (performanceInfoOptional != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("MoeDistributeDispatchV2", "performanceInfo", "not nullptr",
                                                  "performanceInfo not supported in ccu");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }

    int64_t ydtype = expandXOut->GetDataType();
    if (!is950 || (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0)) { // ccu暂时不支持mc2Context
        getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
            x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, elasticInfoOptional,
            performanceInfoOptional, groupEpBuf, epWorldSize, epRankId, moeExpertNum, groupTpBuf, tpWorldSize, tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlgBuf,
            zeroExpertNum, copyExpertNum, constExpertNum, ydtype, expandXOut, dynamicScalesOut, assistInfoForCombineOut,
            expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize, executor);
    } else {
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
        uint64_t hcclBuffSize = 0;
        const char *opName = "moe_distribute_v2";
        auto aclnnRet = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(groupEp, opName, hcclBuffSize, mc2Context);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV3GetWorkspaceSize(
            mc2Context, x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, elasticInfoOptional,
            performanceInfoOptional, epWorldSize, epRankId, moeExpertNum, hcclBuffSize, tpWorldSize, tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlgBuf,
            zeroExpertNum, copyExpertNum, constExpertNum, ydtype, expandXOut, dynamicScalesOut, assistInfoForCombineOut,
            expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize, executor);
#endif
    }
    SetCommArgs(is950, is910B, commAlg, executor);
    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDistributeDispatchBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    if (is950) {
        void *arg = NnopbaseGetUserHandle(executor);
        uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
        CommType commType = static_cast<CommType>(handleVal);
        if (commType == CommType::AIV) {
            OP_LOGD("enter to the  aclnnInnerMoeDistributeDispatchV2Extend");
            return aclnnInnerMoeDistributeDispatchV3(workspace, workspaceSize, executor, stream); // mte走新模版
        }
    }
#endif
    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
