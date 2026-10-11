/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_bandwidth_test.h"
#include <algorithm>
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnnInner_bandwidth_test.h"

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

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

static bool CheckNotNull(const aclTensor* x, const aclTensor* dstRankId, char* group, aclTensor* y,
                         aclTensor* receiveCnt)
{
    OP_LOGD("aclnn_bandwidth_test CheckNotNull start");
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(dstRankId, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(receiveCnt, return false);
    OP_LOGD("aclnn_bandwidth_test CheckNotNull success");

    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group name is Empty");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* dstRankId, char* group, int64_t worldSize,
                               int64_t maxBs, int64_t mode, aclTensor* y, aclTensor* receiveCnt)
{
    OP_LOGD("aclnn_bandwidth_test CheckParams start");
    CHECK_RET(CheckNotNull(x, dstRankId, group, y, receiveCnt), ACLNN_ERR_PARAM_NULLPTR);

    if (worldSize <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "worldSize must be positive, got %ld", worldSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maxBs <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maxBs must be positive, got %ld", maxBs);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (mode < 0 || mode > 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode must be 0 or 1, got %ld", mode);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group name exceeds max length %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    OP_LOGD("aclnn_bandwidth_test CheckParams success");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBandwidthTestGetWorkspaceSize(const aclTensor* x, const aclTensor* dstRankId, char* group,
                                               int64_t worldSize, int64_t maxBs, int64_t mode, char* commAlg,
                                               int64_t aivNum, aclTensor* y, aclTensor* receiveCnt,
                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnnBandwidthTestGetWorkspaceSize start");
    auto ret_param = CheckParams(x, dstRankId, group, worldSize, maxBs, mode, y, receiveCnt);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    aclnnStatus aclnnRet = ACLNN_ERR_INNER;

#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    aclTensor* mc2Context = nullptr;
    uint64_t hcclBuffSize = 0;
    const char* opName = "bandwidth_test";
    auto ret = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(group, opName, hcclBuffSize, mc2Context);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    aclnnRet =
        aclnnInnerBandwidthTestGetWorkspaceSize(mc2Context, x, dstRankId, group, worldSize, maxBs, mode, hcclBuffSize,
                                                commAlg, aivNum, y, receiveCnt, workspaceSize, executor);
#endif
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }

    OP_LOGD("aclnnBandwidthTestGetWorkspaceSize end");
    return aclnnRet;
}

aclnnStatus aclnnBandwidthTest(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    OP_LOGD("aclnnBandwidthTest start");

    aclnnStatus ret = aclnnInnerBandwidthTest(workspace, workspaceSize, executor, stream);
    OP_LOGD("aclnnBandwidthTest end");
    return ret;
}

#ifdef __cplusplus
}
#endif
