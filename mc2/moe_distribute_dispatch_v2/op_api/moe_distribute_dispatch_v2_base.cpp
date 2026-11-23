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
#include "op_host/util/op_const_def.h"
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

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
extern aclnnStatus aclnnInnerMoeDistributeDispatchV3GetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* expertIds, const aclTensor* scales,
    const aclTensor* xActiveMask, const aclTensor* expertScales, const aclTensor* elasticInfo,
    const aclTensor* performanceInfo, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t cclBufferSize, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t shareExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, int64_t ydtype, int64_t rankSizePerServer,
    aclTensor* expandX, aclTensor* dynamicScales, aclTensor* assist_info_for_combine, aclTensor* expertTokensNums,
    aclTensor* epRecvCounts, aclTensor* tpRecvCounts, aclTensor* expandScales, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerMoeDistributeDispatchV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     aclrtStream stream);
#endif

bool DispatchCheckNotNull(const aclTensor* x, const aclTensor* expertIds, const char* groupEp,
                          [[maybe_unused]] const char* groupTp, aclTensor* expandX,
                          [[maybe_unused]] aclTensor* dynamicScales, aclTensor* assistInfoForCombine,
                          aclTensor* expertTokensNums, aclTensor* epRecvCounts)
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

aclnnStatus DispatchCheckParams(const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
                                const char* groupEp, const char* groupTp, int64_t quantMode, aclTensor* expandX,
                                aclTensor* dynamicScales, aclTensor* assistInfoForCombine, aclTensor* expertTokensNums,
                                aclTensor* epRecvCounts, aclTensor* tpRecvCounts)
{
    (void)tpRecvCounts; // 消除未使用参数告警
    CHECK_RET(DispatchCheckNotNull(x, expertIds, groupEp, groupTp, expandX, dynamicScales, assistInfoForCombine,
                                   expertTokensNums, epRecvCounts),
              ACLNN_ERR_PARAM_NULLPTR);
    constexpr size_t inputDimNum = 2;
    OP_CHECK_WRONG_DIMENSION(x, inputDimNum, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(expertIds, inputDimNum, return ACLNN_ERR_PARAM_INVALID);
    if (x->GetViewShape().GetDim(0) != expertIds->GetViewShape().GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x and expertIds must have the same batch size, but got x shape %s and expertIds shape %s.",
                op::ToString(x->GetViewShape()).GetString(), op::ToString(expertIds->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // groupTp为预留参数(当前版本不支持TP域通信)，nullptr视同空串，避免后续strnlen空指针解引用
    if (groupTp == nullptr) {
        groupTp = "";
    }

    if (quantMode >= DISPATCH_DYNAMIC_QUANT_MODE) {
        OP_LOGD("quantMode = 2, dynamicScales can't be null");
        CHECK_RET(dynamicScales != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (static_cast<int32_t>(x->GetDataType()) == static_cast<int32_t>(ACL_INT32)) {
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

static inline void SafeCopyGroupBuf(char* dst, size_t dstSize, const char* src, size_t maxCopyLen)
{
    if (src != nullptr) {
        (void)strncpy_s(dst, dstSize, src, maxCopyLen);
    }
}

static void SetCommArgs(aclOpExecutor** executor)
{
    if (NnopbaseSetHcclServerType) { // 给ACLnn框架指定通讯方式。
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
}

static aclnnStatus ResolveInnerWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional, const aclTensor* elasticInfoOptional,
    const aclTensor* performanceInfoOptional, const char* groupEp, char* groupEpBuf, char* groupTpBuf,
    const char* commAlg, char* commAlgBuf, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor* expandXOut,
    aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut,
    aclTensor* epRecvCountsOut, aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    aclnnStatus getWorkspaceSizesRes = ACLNN_ERR_INNER;
    int64_t ydtype = expandXOut->GetDataType();
    if (GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_2201) {
        getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
            x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, elasticInfoOptional,
            performanceInfoOptional, groupEpBuf, epWorldSize, epRankId, moeExpertNum, groupTpBuf, tpWorldSize, tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlgBuf,
            zeroExpertNum, copyExpertNum, constExpertNum, ydtype, expandXOut, dynamicScalesOut, assistInfoForCombineOut,
            expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize, executor);
    } else {
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
        aclTensor* mc2Context = nullptr;
        uint64_t hcclBuffSize = 0;
        const char* opName = "moe_distribute_v2";
        uint32_t rankSizePerServer = 0;
        bool isA5Hierarchy = (GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) &&
                             (commAlg != nullptr) && (std::strcmp(commAlg, "hierarchy") == 0);
        const auto protocol = isA5Hierarchy ? CommProtocol::COMM_PROTOCOL_ROCE : CommProtocol::COMM_PROTOCOL_UB_MEM;
        auto aclnnRet = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(groupEp, opName, hcclBuffSize, mc2Context, protocol,
                                                                  &rankSizePerServer);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV3GetWorkspaceSize(
            mc2Context, x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, elasticInfoOptional,
            performanceInfoOptional, epWorldSize, epRankId, moeExpertNum, hcclBuffSize, tpWorldSize, tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlgBuf,
            zeroExpertNum, copyExpertNum, constExpertNum, ydtype, rankSizePerServer, expandXOut, dynamicScalesOut,
            assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, expandScalesOut,
            workspaceSize, executor);
#endif
    }
    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSizeBase(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional, const aclTensor* elasticInfoOptional,
    const aclTensor* performanceInfoOptional, const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType,
    int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
    int64_t expertTokenNumsType, const char* commAlg, int64_t zeroExpertNum, int64_t copyExpertNum,
    int64_t constExpertNum, aclTensor* expandXOut, aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut,
    aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut, aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("enter aclnnMoeDistributeDispatchGetWorkspaceSizeBase\n");
    auto retParam =
        DispatchCheckParams(x, expertIds, scalesOptional, groupEp, groupTp, quantMode, expandXOut, dynamicScalesOut,
                            assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    if ((HCOMM_VERSION_NUM < HCCL_CHANNEL_SUPPORT_VERSION) &&
        (GetCurrentPlatformInfo().GetCurNpuArch() != Ops::Base::DAV_2201)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("MoeDistributeDispatchV3", "HCOMM_VERSION_NUM",
                                              std::to_string(HCOMM_VERSION_NUM).c_str(),
                                              std::string("should be >= ")
                                                  .append(std::to_string(HCCL_CHANNEL_SUPPORT_VERSION))
                                                  .append(" for A5 and later chip version")
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    char groupEpBuf[HCCL_GROUP_NAME_MAX] = {0};
    SafeCopyGroupBuf(groupEpBuf, HCCL_GROUP_NAME_MAX, groupEp, HCCL_GROUP_NAME_MAX - 1);
    char groupTpBuf[HCCL_GROUP_NAME_MAX] = {0}; // TP group is unused.
    char commAlgBuf[HCCL_GROUP_NAME_MAX] = {0};
    SafeCopyGroupBuf(commAlgBuf, HCCL_GROUP_NAME_MAX, commAlg, HCCL_GROUP_NAME_MAX - 1);

    auto getWorkspaceSizesRes = ResolveInnerWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, elasticInfoOptional,
        performanceInfoOptional, groupEp, groupEpBuf, groupTpBuf, commAlg, commAlgBuf, epWorldSize, epRankId,
        moeExpertNum, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs,
        expertTokenNumsType, zeroExpertNum, copyExpertNum, constExpertNum, expandXOut, dynamicScalesOut,
        assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize,
        executor);
    CHECK_RET(getWorkspaceSizesRes == ACLNN_SUCCESS, getWorkspaceSizesRes);
    SetCommArgs(executor);
    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDistributeDispatchBase(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    if (GetCurrentPlatformInfo().GetCurNpuArch() != Ops::Base::DAV_2201) {
        OP_LOGD("enter to the  aclnnInnerMoeDistributeDispatchV2Extend");
        // david走v3
        return aclnnInnerMoeDistributeDispatchV3(workspace, workspaceSize, executor, stream);
    }
#endif
    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
