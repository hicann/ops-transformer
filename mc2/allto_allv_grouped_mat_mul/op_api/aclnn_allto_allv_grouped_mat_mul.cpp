/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_allto_allv_grouped_mat_mul.h"
#include <algorithm>
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_host/util/op_const_def.h"
#include "mc2_log_compat.h"
#include "opdev/common_types.h"
#include "aclnnInner_allto_allv_grouped_mat_mul.h"
#include "mc2_comm_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

// check nullptr
static bool CheckNullStatus(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *mmXOptional,
                            const aclTensor *mmWeightOptional, const char *group, bool permuteOutFlag, aclTensor *gmmY,
                            const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(gmmX, return false);
    OP_CHECK_NULL(gmmWeight, return false);
    OP_CHECK_NULL(gmmY, return false);
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvGroupedMatMulGetWorkspaceSize", "group");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAlltoAllvGroupedMatMulGetWorkspaceSize",
                                              "mmXOptional/mmWeightOptional/mmYOptional", "mixed null/non-null",
                                              "should all be null or all not be null");
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAlltoAllvGroupedMatMulGetWorkspaceSize", "permuteOutFlag",
                                              "mismatched", "Optional output flag does not match optional output ptr");
        return false;
    }
    return true;
}

static aclnnStatus CheckOptionalCountTensors(const aclTensor *sendCountsTensorOptional,
                                             const aclTensor *recvCountsTensorOptional)
{
    if (sendCountsTensorOptional == nullptr && recvCountsTensorOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "sendCountsTensorOptional and recvCountsTensorOptional are unsupported and must both be nullptr.");
    return ACLNN_ERR_PARAM_INVALID;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight,
                               const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                               const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const char *group,
                               int64_t epWorldSize, bool permuteOutFlag, aclTensor *gmmY, aclTensor *mmYOptional,
                               aclTensor *permuteOutOptional)
{
    (void)epWorldSize; // Unused
    const aclnnStatus optionalCountRet = CheckOptionalCountTensors(sendCountsTensorOptional, recvCountsTensorOptional);
    CHECK_RET(optionalCountRet == ACLNN_SUCCESS, optionalCountRet);
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, mmXOptional, mmWeightOptional, group, permuteOutFlag, gmmY, mmYOptional,
                              permuteOutOptional),
              ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnAlltoAllvGroupedMatMulGetWorkspaceSize", "group", "too long",
            ("group name length must be less than " + std::to_string(HCCL_GROUP_NAME_MAX)).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSendAndRecv(const aclIntArray *sendCounts, const aclIntArray *recvCounts)
{
    if (sendCounts == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvGroupedMatMulGetWorkspaceSize", "sendCounts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (recvCounts == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvGroupedMatMulGetWorkspaceSize", "recvCounts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t recvSize = 0U; // recvCounts的大小
    uint64_t sendSize = 0U; // recvCounts的大小
    aclGetIntArraySize(recvCounts, &recvSize);
    aclGetIntArraySize(sendCounts, &sendSize);
    if (recvSize == 0U) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvGroupedMatMulGetWorkspaceSize", "recvCounts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sendSize == 0U) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvGroupedMatMulGetWorkspaceSize", "sendCounts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, aclTensor *gmmY, aclTensor *mmYOptional,
    aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto ret_param =
        CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                    group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);
    const char *commMode = "ai_cpu";
    char *str_commMode = const_cast<char *>(commMode);
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        const_cast<char *>(group), epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag,
        str_commMode, gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllvGroupedMatmul, aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize ret %d.", ret);
    if (*executor != nullptr) {
        uint8_t commModeType = Mc2Comm::COMM_MODE_AICPU;
        void *args = reinterpret_cast<void *>(static_cast<uint8_t>(commModeType));
        NnopbaseSetUserHandle(*executor, args);
    }
    return ret;
}

aclnnStatus aclnnAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_AICPU) {
                OP_LOGD("arch35, use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                OP_LOGD("arch35, use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        } else {
            OP_LOGD("Arch22 platform, use AICPU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMul(workspace, workspaceSize, executor, stream);
    OP_LOGD("AlltoAllvGroupedMatmul, aclnnInnerAlltoAllvGroupedMatMul ret %d.", ret);
    return ret;
}

#ifdef __cplusplus
}
#endif
