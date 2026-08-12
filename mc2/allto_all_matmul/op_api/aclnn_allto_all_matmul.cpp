/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_allto_all_matmul.h"

#include "allto_all_matmul_base.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

// 两段式接口
extern "C" aclnnStatus aclnnAlltoAllMatmulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                           const aclTensor *biasOptional,
                                                           const aclIntArray *alltoAllAxesOptional, const char *group,
                                                           bool transposeX1, bool transposeX2, const aclTensor *output,
                                                           const aclTensor *alltoAllOutOptional,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnAlltoAllMatmulGetWorkspaceSize start");

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    const char *commMode =
        (npuArch == NpuArch::DAV_3510 || socVersion == op::SocVersion::ASCEND910_93) ? "ai_cpu" : "aiv";

    return aclnnAlltoAllMatmulBaseGetWorkspaceSize(x1, x2, biasOptional, alltoAllAxesOptional, group, commMode,
                                                   transposeX1, transposeX2, output, alltoAllOutOptional, workspaceSize,
                                                   executor);
}

extern "C" aclnnStatus aclnnAlltoAllMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    OP_LOGD("aclnnAlltoAllMatmul start");
    return aclnnAlltoAllMatmulBase(workspace, workspaceSize, executor, stream);
}
