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
 * \file aclnn_matmul_all_reduce_v2.cpp
 * \brief
 */
#include "aclnn_matmul_all_reduce_v2.h"

#include "aclnnInner_matmul_all_reduce.h"
#include "matmul_all_reduce_util.h"
#include "common/utils/hccl_util.h"
#include "matmul_all_reduce_util.h"
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

static aclnnStatus InnerMatmulAllReduceV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                          const aclTensor *bias, const aclTensor *x3, const char *group,
                                                          const char *reduceOp, int64_t commTurn,
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
    const char *commModePtr = "ai_cpu";
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, x2, bias, x3, scale, offset, dequantScale, pertokenScale, commQuantScale1, commQuantScale2,
        const_cast<char *>(group), const_cast<char *>(reduceOp), transposeX1, transposeX2, commTurn, antiquantGroupSize,
        0, yDtype, 0, const_cast<char *>(commModePtr), output, workspaceSize, executor);
    OP_LOGD("MatmulAllReduceV2, aclnnMatmulAllReduceGetWorkspaceSize ret %d", ret);

    OP_LOGI("Group %s, reduce op %s, trans flag %u %u, ret %d.", group, reduceOp, transposeX1, transposeX2, ret);
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

aclnnStatus aclnnMatmulAllReduceV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                   const aclTensor *x3, const char *group, const char *reduceOp,
                                                   int64_t commTurn, int64_t streamMode, const aclTensor *output,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = MatmulAllReduceCheckParams(x1, x2, x3, bias, reduceOp, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    aclnnStatus ret = InnerMatmulAllReduceV2GetWorkspaceSize(x1, x2, bias, x3, group, reduceOp, commTurn, output,
                                                             workspaceSize, executor);
    OP_LOGD("MatmulAllReduceV2, end ret %d", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnMatmulAllReduceV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
    OP_API_CHECK(ret != ACLNN_SUCCESS, {
        OP_LOGE_LIBOPAPI_REPORT("aclnnMatmulAllReduceV2", "MatmulAllReduceV2, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    });
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
