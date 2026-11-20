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
#include <cstring>

#include "aclnn_distribute_barrier_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "aclnnInner_distribute_barrier.h"
#include "distribute_barrier_base.h"
#include "op_host/util/op_const_def.h"

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM (HCCL_CHANNEL_SUPPORT_VERSION)
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#include "aclnnInner_distribute_barrier_extend.h"
#endif

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

// check nullptr
bool BarrierCheckNullStatus(const aclTensor* xRef, const char* group)
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
aclnnStatus BarrierCheckParams(const aclTensor* xRef, const char* group)
{
    CHECK_RET(BarrierCheckNullStatus(xRef, group), ACLNN_ERR_PARAM_NULLPTR);
    auto groupStrnLen = strnlen(group, HCCL_GROUP_NAME_MAX);
    if ((groupStrnLen >= HCCL_GROUP_NAME_MAX) || (groupStrnLen == 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnDistributeBarrier", "group", std::to_string(groupStrnLen).c_str(),
                                              "The value of group must be in the range (0, HCCL_GROUP_NAME_MAX)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool BarrierIsA5()
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return npuArch == Ops::Base::DAV_3510 || npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202;
}

bool BarrierSupportsStaticGraph(const char* group, int64_t worldSize)
{
    if (!BarrierIsA5()) {
        return true;
    }
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    return Mc2Aclnn::Mc2Context::SupportsBarrierStaticGraph(group, worldSize);
#else
    return false;
#endif
}

aclnnStatus aclnnDistributeBarrierGetWorkspaceSizeBase(const aclTensor* xRef, const aclTensor* timeOut,
                                                       const aclTensor* elasticInfo, const char* group,
                                                       int64_t worldSize, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnDistributeBarrier", workspaceSize == nullptr ? "workspaceSize" : "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto retParam = BarrierCheckParams(xRef, group);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    char groupBuf[HCCL_GROUP_NAME_MAX] = {0};
    (void)strncpy_s(groupBuf, HCCL_GROUP_NAME_MAX, group, HCCL_GROUP_NAME_MAX - 1);
    if (!BarrierIsA5()) {
        return aclnnInnerDistributeBarrierGetWorkspaceSize(const_cast<aclTensor*>(xRef), timeOut, elasticInfo, groupBuf,
                                                           worldSize, workspaceSize, executor);
    }
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    aclTensor* mc2Context = nullptr;
    aclnnStatus ret = Mc2Aclnn::Mc2Context::GetBarrierContextTensor(group, worldSize, mc2Context);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return aclnnInnerDistributeBarrierExtendGetWorkspaceSize(mc2Context, const_cast<aclTensor*>(xRef), timeOut,
                                                             elasticInfo, groupBuf, worldSize, workspaceSize, executor);
#else
    OP_LOGE_LIBOPAPI_REPORT(
        "DistributeBarrier", "A5 barrier requires HCOMM channel support, compiled version=%u, minimum=%u",
        static_cast<uint32_t>(HCOMM_VERSION_NUM), static_cast<uint32_t>(HCCL_CHANNEL_SUPPORT_VERSION));
    return ACLNN_ERR_INNER;
#endif
}

aclnnStatus aclnnDistributeBarrierBase(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    if (BarrierIsA5()) {
        OP_LOGD("aclnn_distribute_barrier_extend inner start");
        return aclnnInnerDistributeBarrierExtend(workspace, workspaceSize, executor, stream);
    }
#endif
    return aclnnInnerDistributeBarrier(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
