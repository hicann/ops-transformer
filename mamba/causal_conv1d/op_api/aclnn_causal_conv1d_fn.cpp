/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <tuple>
#include <cstddef>
#include <cstring>
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "causal_conv1d.h"
#include "aclnn_causal_conv1d_fn.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static int64_t ParseActivationStr(const char *str)
{
    if (str == nullptr || std::strcmp(str, "") == 0 || std::strcmp(str, "silu") == 0) {
        return 1;
    }
    if (std::strcmp(str, "none") == 0) {
        return 0;
    }
    return -1;
}

static bool CheckNotNull(const aclTensor *x, const aclTensor *weight, const aclTensor *convStatesRef,
                         const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(convStatesRef, return false);
    OP_CHECK_NULL(y, return false);
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "expected a value of type uint64_t* but got null for argument workspaceSize.");
        return false;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "expected a value of type aclOpExecutor** but got null for argument executor.");
        return false;
    }
    return true;
}

static bool CheckDtype(const aclTensor *x, const aclTensor *weight, const aclTensor *convStatesRef,
                       const aclTensor *biasOptional, const aclTensor *queryStartLocOptional,
                       const aclTensor *cacheIndicesOptional, const aclTensor *initialStateModeOptional,
                       const aclTensor *y)
{
    auto xDtype = x->GetDataType();
    if (xDtype != op::DataType::DT_FLOAT16 && xDtype != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dtype must be FLOAT16 or BFLOAT16, but got %s.",
                op::ToString(xDtype).GetString());
        return false;
    }
    if (weight->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight dtype must match x dtype(%s), but got %s.",
                op::ToString(xDtype).GetString(), op::ToString(weight->GetDataType()).GetString());
        return false;
    }
    if (convStatesRef->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "convStatesRef dtype must match x dtype(%s), but got %s.",
                op::ToString(xDtype).GetString(), op::ToString(convStatesRef->GetDataType()).GetString());
        return false;
    }
    if (y->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dtype must match x dtype(%s), but got %s.",
                op::ToString(xDtype).GetString(), op::ToString(y->GetDataType()).GetString());
        return false;
    }
    if (biasOptional != nullptr && biasOptional->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "biasOptional dtype must match x dtype(%s), but got %s.",
                op::ToString(xDtype).GetString(), op::ToString(biasOptional->GetDataType()).GetString());
        return false;
    }
    if (queryStartLocOptional != nullptr && queryStartLocOptional->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "queryStartLocOptional dtype must be INT32, but got %s.",
                op::ToString(queryStartLocOptional->GetDataType()).GetString());
        return false;
    }
    if (cacheIndicesOptional != nullptr && cacheIndicesOptional->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cacheIndicesOptional dtype must be INT32, but got %s.",
                op::ToString(cacheIndicesOptional->GetDataType()).GetString());
        return false;
    }
    if (initialStateModeOptional != nullptr && initialStateModeOptional->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "initialStateModeOptional dtype must be INT32, but got %s.",
                op::ToString(initialStateModeOptional->GetDataType()).GetString());
        return false;
    }
    return true;
}

static int64_t GetDtypeSize(const aclTensor *tensor)
{
    auto dtype = tensor->GetDataType();
    if (dtype == op::DataType::DT_FLOAT16 || dtype == op::DataType::DT_BF16) {
        return 2;
    }
    return 0;
}

static bool GetQslSize(const aclTensor *qsl, int64_t &outSize)
{
    if (qsl == nullptr) {
        return false;
    }
    auto qslShape = qsl->GetViewShape();
    if (qslShape.GetDimNum() != 1 || qslShape.GetDim(0) <= 0) {
        return false;
    }
    outSize = qslShape.GetDim(0);
    return true;
}

static bool CheckXShape(const aclTensor *x, int64_t &dim)
{
    auto xShape = x->GetViewShape();
    auto xDimNum = xShape.GetDimNum();
    if (xDimNum != 2 && xDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dim num must be 2 (varlen) or 3 (fixed batch), but got %zuD.", xDimNum);
        return false;
    }
    if (xDimNum == 3) {
        int64_t seqLen = xShape.GetDim(1);
        if (seqLen <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "3D x seqLen must be > 0, but got %ld.", seqLen);
            return false;
        }
    }
    dim = (xDimNum == 3) ? xShape.GetDim(2) : xShape.GetDim(1);
    return true;
}

static bool CheckWeightShape(const aclTensor *weight, int64_t dim, int64_t &kernelWidth)
{
    auto wShape = weight->GetViewShape();
    if (wShape.GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight dim num must be 2 [K, dim], but got %zuD.", wShape.GetDimNum());
        return false;
    }
    kernelWidth = wShape.GetDim(0);
    if (kernelWidth < 2 || kernelWidth > 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernel width K must be in [2, 4], but got %ld.", kernelWidth);
        return false;
    }
    if (wShape.GetDim(1) != dim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight dim(%ld) must match x dim(%ld).", wShape.GetDim(1), dim);
        return false;
    }
    return true;
}

static bool CheckConvStatesShape(const aclTensor *convStatesRef, int64_t dim, int64_t kernelWidth)
{
    const int64_t DIM_MIN = 64;
    const int64_t DIM_MAX = 16384;
    const int64_t DIM_ALIGN_BYTES = 32;
    auto csShape = convStatesRef->GetViewShape();
    if (csShape.GetDimNum() != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "convStatesRef dim num must be 3 [numCacheLines, stateLen, dim], but got %zuD.",
                csShape.GetDimNum());
        return false;
    }
    if (csShape.GetDim(2) != dim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "convStatesRef dim(%ld) must match x dim(%ld).", csShape.GetDim(2), dim);
        return false;
    }
    int64_t stateLen = csShape.GetDim(1);
    if (stateLen < kernelWidth - 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "convStatesRef stateLen(%ld) must be >= K-1(%ld).", stateLen, kernelWidth - 1);
        return false;
    }
    if (dim < DIM_MIN || dim > DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must be in [%ld, %ld], but got %ld.", DIM_MIN, DIM_MAX, dim);
        return false;
    }
    int64_t dtypeSize = GetDtypeSize(convStatesRef);
    if (dtypeSize == 0 || (dim * dtypeSize) % DIM_ALIGN_BYTES != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "dim(%ld) * dtypeSize(%ld) must be %ld-byte aligned.", dim, dtypeSize, DIM_ALIGN_BYTES);
        return false;
    }
    return true;
}

static bool CheckScenarioConstraints(const aclTensor *x, const aclTensor *convStatesRef,
                                     const aclTensor *queryStartLocOptional)
{
    const int64_t BATCH_MIN = 1;
    const int64_t BATCH_MAX = 1024;
    auto xShape = x->GetViewShape();
    auto xDimNum = xShape.GetDimNum();

    int64_t qslSize = 0;
    bool qslPresent = GetQslSize(queryStartLocOptional, qslSize);

    int64_t batch = 0;
    if (xDimNum == 2) {
        if (!qslPresent) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "2D x (varlen) requires queryStartLocOptional to be provided.");
            return false;
        }
        batch = qslSize - 1;
    } else {
        batch = xShape.GetDim(0);
        if (qslPresent && qslSize != batch + 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "queryStartLocOptional size(%ld) must equal batch+1(%ld) for 3D x.", qslSize, batch + 1);
            return false;
        }
    }

    if (batch < BATCH_MIN || batch > BATCH_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch must be in [%ld, %ld], but got %ld.", BATCH_MIN, BATCH_MAX, batch);
        return false;
    }
    int64_t numCacheLines = convStatesRef->GetViewShape().GetDim(0);
    if (numCacheLines < batch) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "convStatesRef numCacheLines(%ld) must be >= batch(%ld).", numCacheLines, batch);
        return false;
    }
    return true;
}

static bool CheckOutputShape(const aclTensor *x, const aclTensor *y)
{
    auto xDimNum = x->GetViewShape().GetDimNum();
    auto yDimNum = y->GetViewShape().GetDimNum();
    if (yDimNum != xDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dim num(%zuD) must match x dim num(%zuD).", yDimNum, xDimNum);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *x, const aclTensor *weight, const aclTensor *convStatesRef,
                       const aclTensor *queryStartLocOptional, const aclTensor *y)
{
    int64_t dim = 0;
    CHECK_RET(CheckXShape(x, dim), false);

    int64_t kernelWidth = 0;
    CHECK_RET(CheckWeightShape(weight, dim, kernelWidth), false);

    CHECK_RET(CheckConvStatesShape(convStatesRef, dim, kernelWidth), false);

    CHECK_RET(CheckScenarioConstraints(x, convStatesRef, queryStartLocOptional), false);

    CHECK_RET(CheckOutputShape(x, y), false);
    return true;
}

aclnnStatus CausalConv1dFnCommonProcess(
    const aclTensor *x, const aclTensor *weight, aclTensor *convStatesRef, const aclTensor *biasOptional,
    const aclTensor *queryStartLocOptional, const aclTensor *cacheIndicesOptional,
    const aclTensor *initialStateModeOptional, const aclTensor *blockIdxFirstScheduledTokenOptional,
    const aclTensor *blockIdxLastScheduledTokenOptional, const aclTensor *initialStateIdxOptional,
    const aclTensor *numComputedTokensOptional, const char *activation, int64_t nullBlockId, int64_t blockSizeToAlign,
    aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    CHECK_RET(CheckNotNull(x, weight, convStatesRef, y, workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtype(x, weight, convStatesRef, biasOptional, queryStartLocOptional, cacheIndicesOptional,
                         initialStateModeOptional, y),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(x, weight, convStatesRef, queryStartLocOptional, y), ACLNN_ERR_PARAM_INVALID);

    const aclTensor *xFinal = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(xFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous x failed.");

    aclTensor *convStatesFinal = const_cast<aclTensor *>(l0op::Contiguous(convStatesRef, uniqueExecutor.get()));
    CHECK_COND(convStatesFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous convStatesRef failed.");

    weight = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_COND(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous weight failed.");

    if (biasOptional != nullptr) {
        biasOptional = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_COND(biasOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous biasOptional failed.");
    }
    if (cacheIndicesOptional != nullptr) {
        cacheIndicesOptional = l0op::Contiguous(cacheIndicesOptional, uniqueExecutor.get());
        CHECK_COND(cacheIndicesOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous cacheIndicesOptional failed.");
    }
    if (queryStartLocOptional != nullptr) {
        queryStartLocOptional = l0op::Contiguous(queryStartLocOptional, uniqueExecutor.get());
        CHECK_COND(queryStartLocOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "Contiguous queryStartLocOptional failed.");
    }
    if (initialStateModeOptional != nullptr) {
        initialStateModeOptional = l0op::Contiguous(initialStateModeOptional, uniqueExecutor.get());
        CHECK_COND(initialStateModeOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "Contiguous initialStateModeOptional failed.");
    }

    CHECK_COND(ParseActivationStr(activation) >= 0, ACLNN_ERR_PARAM_INVALID, "Invalid activation: %s",
               activation ? activation : "null");

    bool ok =
        l0op::CausalConv1d(xFinal, weight, convStatesFinal, biasOptional, queryStartLocOptional, cacheIndicesOptional,
                           initialStateModeOptional, nullptr, activation, nullBlockId, y, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

ACLNN_API aclnnStatus aclnnCausalConv1dFnGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, aclTensor *convStatesRef, const aclTensor *biasOptional,
    const aclTensor *queryStartLocOptional, const aclTensor *cacheIndicesOptional,
    const aclTensor *initialStateModeOptional, const aclTensor *blockIdxFirstScheduledTokenOptional,
    const aclTensor *blockIdxLastScheduledTokenOptional, const aclTensor *initialStateIdxOptional,
    const aclTensor *numComputedTokensOptional, const char *activation, int64_t nullBlockId, int64_t blockSizeToAlign,
    aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnCausalConv1dFn,
                   DFX_IN(x, weight, convStatesRef, biasOptional, queryStartLocOptional, cacheIndicesOptional,
                          initialStateModeOptional),
                   DFX_OUT(convStatesRef, y));
    return CausalConv1dFnCommonProcess(
        x, weight, convStatesRef, biasOptional, queryStartLocOptional, cacheIndicesOptional, initialStateModeOptional,
        blockIdxFirstScheduledTokenOptional, blockIdxLastScheduledTokenOptional, initialStateIdxOptional,
        numComputedTokensOptional, activation, nullBlockId, blockSizeToAlign, y, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnCausalConv1dFn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCausalConv1dFn);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
