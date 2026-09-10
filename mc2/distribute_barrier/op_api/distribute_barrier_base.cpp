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
 * \file distribute_barrier_base.cpp
 * \brief
 */
#include <algorithm>

#include "aclnn_distribute_barrier_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "aclnnInner_distribute_barrier.h"
#include "distribute_barrier_base.h"

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM (HCCL_CHANNEL_SUPPORT_VERSION)
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#endif

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern aclnnStatus aclnnInnerDistributeBarrierExtend(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream);

extern aclnnStatus aclnnInnerDistributeBarrierExtendGetWorkspaceSize(const aclTensor *context, const aclTensor *xRef,
                                                                     const aclTensor *timeOut,
                                                                     const aclTensor *elasticInfo, const char *group,
                                                                     int64_t worldSize, uint64_t *workspaceSize,
                                                                     aclOpExecutor **executor);

// check nullptr
bool BarrierCheckNullStatus(const aclTensor *xRef, const char *group)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(xRef, return false);
    if (group == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnDistributeBarrier", "group");
        return false;
    }
    return true;
}

// 入参校验
aclnnStatus BarrierCheckParams(const aclTensor *xRef, const char *group)
{
    CHECK_RET(BarrierCheckNullStatus(xRef, group), ACLNN_ERR_PARAM_NULLPTR);
    auto groupStrnLen = strnlen(group, HCCL_GROUP_NAME_MAX);
    if ((groupStrnLen >= HCCL_GROUP_NAME_MAX) || (groupStrnLen == 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnDistributeBarrier", "group", std::to_string(groupStrnLen).c_str(),
                                              "The value of group must be in the range (0, HCCL_GROUP_NAME_MAX)");
        return false;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDistributeBarrierGetWorkspaceSizeBase(const aclTensor *xRef, const aclTensor *timeOut,
                                                       const aclTensor *elasticInfo, const char *group,
                                                       int64_t worldSize, uint64_t *workspaceSize,
                                                       aclOpExecutor **executor)
{
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    auto retParam = BarrierCheckParams(xRef, group);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclTensor *mc2Context = nullptr;
    aclnnStatus getWorkspaceSizesRes = ACLNN_ERR_INNER;
    aclnnStatus ret = ACLNN_ERR_INNER;

    char groupBuf[HCCL_GROUP_NAME_MAX] = {0};
    if (group != nullptr) {
        (void)strncpy_s(groupBuf, HCCL_GROUP_NAME_MAX, group, HCCL_GROUP_NAME_MAX - 1);
    }
    aclTensor *xRefBuf = const_cast<aclTensor *>(xRef);
    if (!is950) {
        getWorkspaceSizesRes = aclnnInnerDistributeBarrierGetWorkspaceSize(xRefBuf, timeOut, elasticInfo, groupBuf,
                                                                           worldSize, workspaceSize, executor);
    } else {
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
        uint64_t hcclBuffSize = 0;
        const char *opName = "distribute_barrier_extend";
        ret = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(group, opName, hcclBuffSize, mc2Context);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        getWorkspaceSizesRes = aclnnInnerDistributeBarrierExtendGetWorkspaceSize(
            mc2Context, xRefBuf, timeOut, elasticInfo, groupBuf, worldSize, workspaceSize, executor);
#endif
    }
    return getWorkspaceSizesRes;
}

aclnnStatus aclnnDistributeBarrierBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    if (is950) {
        OP_LOGD("aclnn_distribute_barrier_extend inner start");
        return aclnnInnerDistributeBarrierExtend(workspace, workspaceSize, executor, stream);
    }
#endif
    return aclnnInnerDistributeBarrier(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
