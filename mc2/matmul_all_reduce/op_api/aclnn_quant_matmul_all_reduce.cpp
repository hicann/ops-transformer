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
 * \file aclnn_quant_matmul_all_reduce.cpp
 * \brief
 */
#include "aclnn_quant_matmul_all_reduce.h"

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
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                      const aclTensor *x3, const aclTensor *dequantScale,
                                                      const char *group, const char *reduceOp, int64_t commTurn,
                                                      int64_t streamMode, const aclTensor *output,
                                                      uint64_t *workspaceSize, aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam =
        QuantMatmulAllReduceCheckParams(x1, x2, bias, dequantScale, nullptr, x3, reduceOp, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // dequantScale转为uint64
    auto dequant = const_cast<aclTensor *>(dequantScale);
    if (dequant == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnQuantMatmulAllReduce", "dequantScale");
        return ACLNN_ERR_INNER;
    }
    if (dequant->GetDataType() == op::DataType::DT_INT64) {
        dequant->SetDataType(op::DataType::DT_UINT64);
    }

    aclnnStatus ret = InnerQuantMatmulAllReduceGetWorkspaceSize(x1, x2, bias, x3, dequant, nullptr, group, reduceOp,
                                                                commTurn, output, workspaceSize, executor);
    OP_LOGD("QuantMatmulAllReduce, end ret %d", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnQuantMatmulAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
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
        OP_LOGE_LIBOPAPI_REPORT("aclnnQuantMatmulAllReduce", "QuantMatmulAllReduceLaunchTask fail, ret: %d.", ret);
        return ACLNN_ERR_INNER;
    });
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
