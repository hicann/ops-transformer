/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_allto_allv_grouped_mat_mul_v2.h"
#include <algorithm>
#include <cstring>
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_host/util/op_const_def.h"
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

static constexpr uint32_t RANK_DIM_BOUNDARY = 8;

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

static aclnnStatus CheckAndHandleCommMode(const char *commModeStr, uint8_t &commModeEnum)
{
    if (commModeStr == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Optional commMode name is Empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto arch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (std::strcmp(commModeStr, "ai_cpu") == 0) {
        commModeEnum = Mc2Comm::COMM_MODE_AICPU;
        return ACLNN_SUCCESS;
    }
    if (arch == Ops::Base::DAV_2201 && std::strcmp(commModeStr, "aiv") == 0) {
        commModeEnum = Mc2Comm::COMM_MODE_AIV;
        return ACLNN_SUCCESS;
    }
    if (arch == Ops::Base::DAV_3510 && std::strcmp(commModeStr, "ccu") == 0) {
        commModeEnum = Mc2Comm::COMM_MODE_CCU;
        return ACLNN_SUCCESS;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AlltoAllvGroupedMatMulV2 commMode %s is unsupported on the current platform.",
            commModeStr);
    return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus CheckAivTensor(const aclTensor *tensor, const char *tensorName, bool allowPtaNcl = false)
{
    const auto dtype = tensor->GetDataType();
    if (dtype != op::DataType::DT_BF16 && dtype != op::DataType::DT_FLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s only supports FP16 or BF16 in AIV mode.", tensorName);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto storageFormat = tensor->GetStorageFormat();
    const bool isNd = storageFormat == op::Format::FORMAT_ND;
    // torch_npu describes a rank-3 base-format tensor as NCL when it creates
    // the aclTensor. Its physical storage is still contiguous ND. Accept that
    // representation only for the rank-3 GMM weight used by the PTA path.
    const bool isPtaNcl =
        allowPtaNcl && storageFormat == op::Format::FORMAT_NCL && tensor->GetViewShape().GetDimNum() == 3U;
    if (!isNd && !isPtaNcl) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s only supports ND format, but got format %d.", tensorName,
                static_cast<int32_t>(storageFormat));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

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
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, left: %u, right: %u, "
            "mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is nullptr: %u",
            (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))),
            (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))),
            mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr);
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Optional output flag does not match optional output ptr!");
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
                               bool permuteOutFlag, aclTensor *gmmY, aclTensor *mmYOptional,
                               aclTensor *permuteOutOptional, bool checkAivTensor)
{
    const aclnnStatus optionalCountRet = CheckOptionalCountTensors(sendCountsTensorOptional, recvCountsTensorOptional);
    CHECK_RET(optionalCountRet == ACLNN_SUCCESS, optionalCountRet);
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, mmXOptional, mmWeightOptional, group, permuteOutFlag, gmmY, mmYOptional,
                              permuteOutOptional),
              ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required group name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!checkAivTensor) {
        return ACLNN_SUCCESS;
    }

    aclnnStatus ret = CheckAivTensor(gmmX, "gmmX");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckAivTensor(gmmWeight, "gmmWeight", true);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckAivTensor(gmmY, "gmmY");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (mmXOptional != nullptr) {
        ret = CheckAivTensor(mmXOptional, "mmXOptional");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckAivTensor(mmWeightOptional, "mmWeightOptional");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = CheckAivTensor(mmYOptional, "mmYOptional");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    if (permuteOutOptional != nullptr) {
        ret = CheckAivTensor(permuteOutOptional, "permuteOutOptional");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSendAndRecv(const aclIntArray *sendCounts, const aclIntArray *recvCounts)
{
    if (sendCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (recvCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t recvSize = 0U; // recvCounts的大小
    uint64_t sendSize = 0U; // recvCounts的大小
    aclGetIntArraySize(recvCounts, &recvSize);
    aclGetIntArraySize(sendCounts, &sendSize);
    if (recvSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sendSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const char *group, const char *commMode, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, aclTensor *gmmY,
    aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto ret_param = CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
                                 mmWeightOptional, group, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional, false);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);
    char *str_commMode = const_cast<char *>(commMode);
    uint8_t commModeEnum = Mc2Comm::COMM_MODE_AICPU;
    aclnnStatus checkCommModeRet = CheckAndHandleCommMode(commMode, commModeEnum);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);
    if (commModeEnum == Mc2Comm::COMM_MODE_AIV) {
        ret_param = CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
                                mmWeightOptional, group, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional, true);
        CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    }
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        const_cast<char *>(group), epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag,
        str_commMode, gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllvGroupedMatmul, aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize ret %d.", ret);
    if (*executor != nullptr) {
        void *args = reinterpret_cast<void *>(static_cast<uintptr_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, args);
    }
    return ret;
}

aclnnStatus aclnnAlltoAllvGroupedMatMulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        const uintptr_t handleVal = reinterpret_cast<uintptr_t>(NnopbaseGetUserHandle(executor));
        const uint8_t commMode = static_cast<uint8_t>(handleVal);
        if (commMode == Mc2Comm::COMM_MODE_AIV) {
            OP_LOGD("AlltoAllvGroupedMatMulV2 uses AIV/MTE communication mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        } else if (commMode == Mc2Comm::COMM_MODE_CCU &&
                   GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) {
            OP_LOGD("AlltoAllvGroupedMatMulV2 uses CCU communication mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            OP_LOGD("AlltoAllvGroupedMatMulV2 uses AICPU communication mode");
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
