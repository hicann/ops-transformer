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
 * \file aclnn_quant_matmul_allto_all.cpp
 * \brief
 */
#include "aclnn_quant_matmul_allto_all.h"
#include "quant_matmul_allto_all_base.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"

// 两段式接口
extern "C" aclnnStatus aclnnQuantMatmulAlltoAllGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x1Scale,
    const aclTensor *x2Scale, const aclTensor *commScaleOptional, const aclTensor *x1OffsetOptional,
    const aclTensor *x2OffsetOptional, const aclIntArray *alltoAllAxesOptional, const char *group, int64_t x1QuantMode,
    int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t groupSize, bool transposeX1,
    bool transposeX2, const aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnQuantMatmulAlltoAllGetWorkspaceSize start");

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    const char *commMode =
        (npuArch == NpuArch::DAV_3510 || socVersion == op::SocVersion::ASCEND910_93) ? "ai_cpu" : "aiv";

    return aclnnQuantMatmulAlltoAllBaseGetWorkspaceSize(
        x1, x2, biasOptional, x1Scale, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        alltoAllAxesOptional, group, commMode, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, groupSize,
        transposeX1, transposeX2, output, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnQuantMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream)
{
    OP_LOGD("aclnnQuantMatmulAlltoAll start");
    return aclnnQuantMatmulAlltoAllBase(workspace, workspaceSize, executor, stream);
}
