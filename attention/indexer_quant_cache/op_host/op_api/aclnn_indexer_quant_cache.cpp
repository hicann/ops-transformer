/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_indexer_quant_cache.h"
#include "indexer_quant_cache.h"

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

aclnnStatus aclnnIndexerQuantCacheGetWorkspaceSize(
    aclTensor *cacheRef,
    aclTensor *cacheScaleRef,
    const aclTensor *x,
    const aclTensor *slotMapping,
    int64_t quantMode,
    bool roundScale,
    float xScale,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnIndexerQuantCache,
                   DFX_IN(cacheRef, cacheScaleRef,
                          x, slotMapping, quantMode, roundScale, xScale),
                   DFX_OUT());

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(cacheRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(cacheScaleRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(slotMapping, return ACLNN_ERR_PARAM_NULLPTR);

    if (cacheRef->IsEmpty() || x->IsEmpty() || slotMapping->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnIndexerQuantCache does not support empty tensor!");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto cacheView = uniqueExecutor->CreateView(cacheRef,
                                                cacheRef->GetViewShape(),
                                                cacheRef->GetStorageShape(),
                                                cacheRef->GetViewStrides(),
                                                cacheRef->GetViewOffset());
    CHECK_RET(cacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scaleView = uniqueExecutor->CreateView(cacheScaleRef,
                                                cacheScaleRef->GetViewShape(),
                                                cacheScaleRef->GetStorageShape(),
                                                cacheScaleRef->GetViewStrides(),
                                                cacheScaleRef->GetViewOffset());
    CHECK_RET(scaleView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto output = l0op::IndexerQuantCache(
        const_cast<aclTensor *>(cacheView), const_cast<aclTensor *>(scaleView),
        x, slotMapping,
        quantMode, roundScale, xScale,
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexerQuantCache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIndexerQuantCache);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif