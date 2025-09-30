/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul_all_reduce.cpp
 * \brief
 */
#include "aclnn_matmul_all_reduce.h"
#include "securec.h"

#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul_all_reduce_util.h"
#include "aclnn_kernels/contiguous.h"
#include "matmul_all_reduce_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMatmulAllReduceGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3,
    const aclTensor* antiquantScale, const aclTensor* antiquantOffset, const aclTensor* dequantScale,
    const aclTensor* pertokenScale, const aclTensor* commQuantScale1, const aclTensor* commQuantScale2,
    const char* group, const char* reduceOp, bool transposeX1, bool transposeX2, int64_t commTurn,
    int64_t antiquantGroupSize, int64_t groupSize, int64_t yDtype, int64_t commQuantMode, const aclTensor* output,
    uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void* executor, const size_t irIndex);

aclnnStatus aclnnMatmulAllReduceGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const char* group, const char* reduceOp,
    int64_t commTurn, int64_t streamMode, const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = MatmulAllReduceCheckParams(x1, x2, nullptr, bias, reduceOp, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    const size_t x1DimNum = x1->GetOriginalShape().GetDimNum();
    if (x1DimNum < 1 || x1DimNum > THREE_DIMS) {
        return ACLNN_ERR_INNER;
    }

    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2);
    aclTensor* pertokenScale = nullptr;
    aclTensor* commQuantScale1 = nullptr;
    aclTensor* commQuantScale2 = nullptr;
    aclTensor* dequantScale = nullptr;
    aclTensor* x3 = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    int64_t antiquantGroupSize = 0;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, x2, bias, x3, scale, offset, dequantScale, pertokenScale, commQuantScale1, commQuantScale2, group, reduceOp,
        transposeX1, transposeX2, commTurn, antiquantGroupSize, 0, yDtype, 0, output, workspaceSize, executor);
    OP_LOGD("MatmulAllReduce, aclnnMatmulAllReduceGetWorkspaceSize ret %d", ret);

    OP_LOGI("Group %s, reduce op %s, trans flag %u %u, ret %d.", group, reduceOp, transposeX1, transposeX2, ret);
#ifdef MC2_UT
    ret = 0;
#endif
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 3U); // 3 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 4U); // 4 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 6U); // 6 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 8U); // 8 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 9U); // 9 is input irIndex
        }
    }
    OP_LOGD("MatmulAllReduce, end ret %d", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();

    aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
    OP_LOGD("MatmulAllReduce, aclnnMatmulAllReduce ret %d", ret);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "MatmulAllReduce, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
