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
 * \file aclnn_matmul_all_to_all.cpp
 * \brief
 */
#include "aclnn_matmul_allto_all.h"
#include "matmul_allto_all_base.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "op_host/util/op_const_def.h"

// 两段式接口
extern "C" aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                           const aclTensor *biasOptional,
                                                           const aclIntArray *alltoAllAxesOptional, const char *group,
                                                           bool transposeX1, bool transposeX2, const aclTensor *output,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnMatmulAlltoAllGetWorkspaceSize start");

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    const char *commMode =
        (npuArch == Ops::Base::DAV_3510 || socVersion == op::SocVersion::ASCEND910_93) ? "ai_cpu" : "aiv";

    return aclnnMatmulAlltoAllBaseGetWorkspaceSize(x1, x2, biasOptional, alltoAllAxesOptional, group, commMode,
                                                   transposeX1, transposeX2, output, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    OP_LOGD("aclnnMatmulAlltoAll start");
    return aclnnMatmulAlltoAllBase(workspace, workspaceSize, executor, stream);
}
