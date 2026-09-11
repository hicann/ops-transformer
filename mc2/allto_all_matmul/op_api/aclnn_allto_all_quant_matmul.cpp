/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_allto_all_quant_matmul.h"

#include "allto_all_quant_matmul_base.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_host/util/op_const_def.h"

// 两段式接口
extern "C" aclnnStatus aclnnAlltoAllQuantMatmulGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x1ScaleOptional,
    const aclTensor *x2Scale, const aclTensor *commScaleOptional, const aclTensor *x1OffsetOptional,
    const aclTensor *x2OffsetOptional, const char *group, const aclIntArray *alltoAllAxesOptional, int64_t x1QuantMode,
    int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype, int64_t groupSize,
    bool transposeX1, bool transposeX2, const aclTensor *output, const aclTensor *alltoAllOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnAlltoAllQuantMatmulGetWorkspaceSize start");

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    const char *commMode =
        (npuArch == Ops::Base::DAV_3510 || socVersion == op::SocVersion::ASCEND910_93) ? "ai_cpu" : "aiv";

    return aclnnAlltoAllQuantMatmulBaseGetWorkspaceSize(
        x1, x2, biasOptional, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional, group,
        commMode, alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
        groupSize, transposeX1, transposeX2, output, alltoAllOutOptional, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnAlltoAllQuantMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream)
{
    OP_LOGD("aclnnAlltoAllQuantMatmul start");
    return aclnnAlltoAllQuantMatmulBase(workspace, workspaceSize, executor, stream);
}
