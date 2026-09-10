/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul_all_reduce_v3.cpp
 * \brief
 */
#include "aclnn_matmul_all_reduce_v3.h"
#include "aclnnInner_matmul_all_reduce.h"
#include "matmul_all_reduce_util.h"
#include "common/utils/hccl_util.h"
#include "mc2_comm_utils.h"
#include "mc2_log_compat.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

static aclnnStatus InnerMatmulAllReduceV3GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                          const aclTensor *bias, const aclTensor *x3, const char *group,
                                                          const char *reduceOp, const char *commMode, int64_t commTurn,
                                                          const aclTensor *output, uint64_t *workspaceSize,
                                                          aclOpExecutor **executor)
{
    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2);
    aclTensor *pertokenScale = nullptr;
    aclTensor *dequantScale = nullptr;
    aclTensor *commQuantScale1 = nullptr;
    aclTensor *commQuantScale2 = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;
    int64_t antiquantGroupSize = 0;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, x2, bias, x3, scale, offset, dequantScale, pertokenScale, commQuantScale1, commQuantScale2,
        const_cast<char *>(group), const_cast<char *>(reduceOp), transposeX1, transposeX2, commTurn, antiquantGroupSize,
        0, yDtype, 0, const_cast<char *>(commMode), output, workspaceSize, executor);
    OP_LOGD("MatmulAllReduceV3, aclnnMatmulAllReduceGetWorkspaceSize ret %d", ret);
    if (ret == ACLNN_SUCCESS && executor != nullptr && *executor != nullptr && commMode != nullptr) {
        // SetUserHandle to pass comm_mode to aclnnMatmulAllReduceV3
        uint8_t commModeEnum = 0;
        if (strcmp(commMode, COMM_MODE_AICPU) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_AICPU;
        } else if (strcmp(commMode, COMM_MODE_CCU) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_CCU;
        } else {
            OP_LOGE_WITH_INVALID_ATTR("InnerMatmulAllReduceV3GetWorkspaceSize", "commMode", commMode,
                                      "'ccu' or 'ai_cpu'");
            return ACLNN_ERR_PARAM_INVALID;
        }
        void *arg = reinterpret_cast<void *>(static_cast<uintptr_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, arg);
    }
#ifdef MC2_UT
    ret = 0;
#endif
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr && executor != nullptr && *executor != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 4U); // 4 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 6U); // 6 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 8U); // 8 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 9U); // 9 is input irIndex
        }
    }
    return ret;
}

aclnnStatus aclnnMatmulAllReduceV3GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                   const aclTensor *x3, const char *group, const char *reduceOp,
                                                   const char *commMode, int64_t commTurn, int64_t streamMode,
                                                   const aclTensor *output, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor)
{
    uint64_t timeStampVal = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = MatmulAllReduceCheckParams(x1, x2, x3, bias, reduceOp, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 检查commMode是否合法
    CHECK_RET(IsCommModeValid(commMode), ACLNN_ERR_PARAM_INVALID);

    aclnnStatus ret = InnerMatmulAllReduceV3GetWorkspaceSize(x1, x2, bias, x3, group, reduceOp, commMode, commTurn,
                                                             output, workspaceSize, executor);
    OP_LOGD("MatmulAllReduceV3, end ret %d", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStampVal, dfxId);
    return ret;
}

aclnnStatus aclnnMatmulAllReduceV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream)
{
    uint64_t timeStampVal = NnopbaseMsprofSysTime();
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Param executor is nullptr.");
        return ACLNN_ERR_INNER;
    }
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_CCU) {
                OP_LOGD("A5 aclnnMatmulAllReduceV3: NnopbaseHcclServerType, use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            } else {
                OP_LOGD("A5 aclnnMatmulAllReduceV3: NnopbaseHcclServerType, use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            }
        } else {
            OP_LOGD("A2 aclnnMatmulAllReduceV3: NnopbaseHcclServerType, use AICPU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
    OP_API_CHECK(ret != ACLNN_SUCCESS, {
        OP_LOGE(ACLNN_ERR_INNER, "MatmulAllReduceV3, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    });
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStampVal, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
