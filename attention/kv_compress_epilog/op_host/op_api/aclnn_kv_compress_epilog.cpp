/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kv_compress_epilog.h"
#include "kv_compress_epilog.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(
    aclTensor *cacheRef,
    const aclTensor *x,
    const aclTensor *slotMapping,
    int64_t quantGroupSize,
    int64_t quantMode,
    bool roundScale,
    float xScale,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnKvCompressEpilog,
                   DFX_IN(cacheRef,
                          x, slotMapping, quantGroupSize, quantMode, roundScale, xScale),
                   DFX_OUT());

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(cacheRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(slotMapping, return ACLNN_ERR_PARAM_NULLPTR);

    if (cacheRef->IsEmpty() || x->IsEmpty() || slotMapping->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnKvCompressEpilog does not support empty tensor!");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto kvCacheView = uniqueExecutor->CreateView(cacheRef,
                                                  cacheRef->GetViewShape(),
                                                  cacheRef->GetStorageShape(),
                                                  cacheRef->GetViewStrides(),
                                                  cacheRef->GetViewOffset());
    CHECK_RET(kvCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto output = l0op::KvCompressEpilog(
        const_cast<aclTensor *>(kvCacheView),
        x, slotMapping,
        quantGroupSize, quantMode, roundScale, xScale,
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvCompressEpilog(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKvCompressEpilog);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif