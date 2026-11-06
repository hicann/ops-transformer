/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_text_gated_residual.h"
#include "text_gated_residual.h"

#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "platform/soc_spec.h"

namespace {
constexpr int64_t HC_ACTIVATION_RANK = 3;
constexpr int64_t HC_WEIGHT_RANK = 2;
constexpr int64_t HC_FEATURE_DIM = 2;
constexpr int64_t HC_INPUT_ELEMENT_BYTES = sizeof(uint16_t);
constexpr int64_t HC_ACCUMULATOR_BYTES = sizeof(float);
} // namespace

using namespace op;

namespace {
bool CheckNotNull(const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp, const aclTensor* mixed)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(wDown, return false);
    OP_CHECK_NULL(wUp, return false);
    OP_CHECK_NULL(mixed, return false);
    return true;
}

aclnnStatus CheckParams(const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp, const aclTensor* mixed,
                        const aclTensor* wGate, bool useCombine, int64_t channels, bool needBackward,
                        const aclTensor* invRmsOptional, const aclTensor* downPreActOptional,
                        const aclTensor* upPreActOptional)
{
    CHECK_RET(CheckNotNull(x, wDown, wUp, mixed), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_COND(channels > 0, ACLNN_ERR_PARAM_INVALID, "channels must be > 0.");
    CHECK_COND(x->GetViewShape().GetDimNum() == HC_ACTIVATION_RANK, ACLNN_ERR_PARAM_INVALID,
               "x must be rank-3 [B,S,D].");
    CHECK_COND(wDown->GetViewShape().GetDimNum() == HC_WEIGHT_RANK, ACLNN_ERR_PARAM_INVALID,
               "w_down must be rank-2 [R,D].");
    CHECK_COND(wUp->GetViewShape().GetDimNum() == HC_WEIGHT_RANK, ACLNN_ERR_PARAM_INVALID,
               "w_up must be rank-2 [D,R].");
    const int64_t b = x->GetViewShape().GetDim(0);
    const int64_t s = x->GetViewShape().GetDim(1);
    const int64_t d = x->GetViewShape().GetDim(HC_FEATURE_DIM);
    const int64_t r = wDown->GetViewShape().GetDim(0);
    CHECK_COND(b >= 0 && s >= 0 && d > 0 && r > 0, ACLNN_ERR_PARAM_INVALID, "invalid nonpositive dimensions.");
    CHECK_COND(b <= UINT32_MAX && s <= UINT32_MAX && d <= UINT32_MAX && r <= UINT32_MAX && channels <= UINT32_MAX,
               ACLNN_ERR_PARAM_INVALID, "dimensions exceed tiling representation.");
    CHECK_COND(s == 0 || b <= UINT32_MAX / s, ACLNN_ERR_PARAM_INVALID, "B*S exceeds tiling representation.");
    const int64_t m = b * s; // B*S was checked against UINT32_MAX above.
    CHECK_COND(m <= INT64_MAX / (d * HC_ACCUMULATOR_BYTES) && m <= INT64_MAX / (r * HC_ACCUMULATOR_BYTES) &&
                   r <= INT64_MAX / (d * HC_INPUT_ELEMENT_BYTES) &&
                   channels <= INT64_MAX / (d * HC_INPUT_ELEMENT_BYTES),
               ACLNN_ERR_PARAM_INVALID, "tensor byte offsets exceed signed 64-bit representation.");
    CHECK_COND(d % channels == 0, ACLNN_ERR_PARAM_INVALID, "D must be divisible by channels.");
    CHECK_COND(mixed->GetViewShape().GetDimNum() == HC_ACTIVATION_RANK && mixed->GetViewShape().GetDim(0) == b &&
                   mixed->GetViewShape().GetDim(1) == s && mixed->GetViewShape().GetDim(HC_FEATURE_DIM) == d / channels,
               ACLNN_ERR_PARAM_INVALID, "mixed must have shape [B,S,H].");
    CHECK_COND(wDown->GetViewShape().GetDim(1) == d, ACLNN_ERR_PARAM_INVALID, "w_down last dim must equal D.");
    CHECK_COND(wUp->GetViewShape().GetDim(0) == d, ACLNN_ERR_PARAM_INVALID, "w_up first dim must equal D.");
    CHECK_COND(wUp->GetViewShape().GetDim(1) == r, ACLNN_ERR_PARAM_INVALID, "w_up last dim must equal R.");
    if (useCombine) {
        CHECK_COND(wGate != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w_gate is required when w_gate is provided.");
        CHECK_COND(wGate->GetViewShape().GetDimNum() == HC_WEIGHT_RANK, ACLNN_ERR_PARAM_INVALID,
                   "w_gate must be rank-2 [C,D].");
        CHECK_COND(wGate->GetViewShape().GetDim(0) == channels && wGate->GetViewShape().GetDim(1) == d,
                   ACLNN_ERR_PARAM_INVALID, "w_gate must have shape [C,D].");
    }
    if (needBackward) {
        CHECK_COND(upPreActOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "upPreAct is required when needBackward is true.");
        CHECK_COND(upPreActOptional->GetDataType() == x->GetDataType(), ACLNN_ERR_PARAM_INVALID,
                   "upPreAct dtype must match x.");
        CHECK_COND(upPreActOptional->GetViewShape().GetDimNum() == HC_ACTIVATION_RANK, ACLNN_ERR_PARAM_INVALID,
                   "upPreAct must be rank-3 [B,S,D].");
        CHECK_COND(upPreActOptional->GetViewShape().GetDim(0) == b && upPreActOptional->GetViewShape().GetDim(1) == s &&
                       upPreActOptional->GetViewShape().GetDim(HC_FEATURE_DIM) == d,
                   ACLNN_ERR_PARAM_INVALID, "upPreAct shape must be [B,S,D].");
        CHECK_COND(invRmsOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR, "invRms is required when needBackward is true.");
        CHECK_COND(downPreActOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "downPreAct is required when needBackward is true.");
        CHECK_COND(invRmsOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "invRms must be FLOAT32.");
        CHECK_COND(downPreActOptional->GetDataType() == x->GetDataType(), ACLNN_ERR_PARAM_INVALID,
                   "downPreAct dtype must match x.");
        CHECK_COND(invRmsOptional->GetViewShape().GetDimNum() == HC_ACTIVATION_RANK, ACLNN_ERR_PARAM_INVALID,
                   "invRms must be rank-3 [B,S,C].");
        CHECK_COND(invRmsOptional->GetViewShape().GetDim(0) == b && invRmsOptional->GetViewShape().GetDim(1) == s &&
                       invRmsOptional->GetViewShape().GetDim(HC_FEATURE_DIM) == channels,
                   ACLNN_ERR_PARAM_INVALID, "invRms shape must be [B,S,C].");
        CHECK_COND(downPreActOptional->GetViewShape().GetDimNum() == HC_ACTIVATION_RANK, ACLNN_ERR_PARAM_INVALID,
                   "downPreAct must be rank-3 [B,S,R].");
        CHECK_COND(downPreActOptional->GetViewShape().GetDim(0) == b &&
                       downPreActOptional->GetViewShape().GetDim(1) == s &&
                       downPreActOptional->GetViewShape().GetDim(HC_FEATURE_DIM) == r,
                   ACLNN_ERR_PARAM_INVALID, "downPreAct shape must be [B,S,R].");
    }
    return ACLNN_SUCCESS;
}

bool IsDav2201()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnTextGatedResidualGetWorkspaceSize(const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp,
                                                   const aclTensor* wGateOptional, const aclTensor* gammaOptional,
                                                   int64_t channels, double eps, bool needBackward,
                                                   const aclTensor* mixed, const aclTensor* injectOptional,
                                                   const aclTensor* invRmsOptional, const aclTensor* downPreActOptional,
                                                   const aclTensor* upPreActOptional, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
    const bool useCombine = wGateOptional != nullptr;
    L2_DFX_PHASE_1(aclnnTextGatedResidual,
                   DFX_IN(x, wDown, wUp, wGateOptional, gammaOptional, channels, eps, needBackward),
                   DFX_OUT(mixed, injectOptional, invRmsOptional, downPreActOptional, upPreActOptional));
    const aclnnStatus paramStatus = CheckParams(x, wDown, wUp, mixed, wGateOptional, useCombine, channels, needBackward,
                                                invRmsOptional, downPreActOptional, upPreActOptional);
    if (paramStatus != ACLNN_SUCCESS) {
        return paramStatus;
    }
    if (gammaOptional != nullptr) {
        CHECK_COND(gammaOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "gamma must be FLOAT32.");
        CHECK_COND(gammaOptional->GetViewShape().GetDimNum() == 1 &&
                       gammaOptional->GetViewShape().GetDim(0) == x->GetViewShape().GetDim(HC_FEATURE_DIM),
                   ACLNN_ERR_PARAM_INVALID, "gamma must have shape [D=C*H].");
    }
    {
        CHECK_COND(std::isfinite(eps) && eps >= 0 && eps <= std::numeric_limits<float>::max(), ACLNN_ERR_PARAM_INVALID,
                   "eps must be >= 0.");
        CHECK_COND(x->GetViewShape().GetDim(HC_FEATURE_DIM) != 0, ACLNN_ERR_PARAM_INVALID, "D must be > 0.");
        const auto xDtype = x->GetDataType();
        CHECK_COND(xDtype == DataType::DT_FLOAT16 || xDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
                   "x dtype must be FLOAT16 or BF16.");
        CHECK_COND(wDown->GetDataType() == xDtype && wUp->GetDataType() == xDtype && mixed->GetDataType() == xDtype,
                   ACLNN_ERR_PARAM_INVALID, "weight and mixed dtype must match x.");
        if (wGateOptional != nullptr) {
            CHECK_COND(wGateOptional->GetDataType() == xDtype, ACLNN_ERR_PARAM_INVALID, "w_gate dtype must match x.");
        }
        if (gammaOptional != nullptr) {
            CHECK_COND(gammaOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                       "gamma must be FLOAT32.");
        }
    }
    if (x->GetViewShape().GetDim(0) == 0) {
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (x->GetViewShape().GetDim(1) == 0) {
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    CHECK_COND(mixed != nullptr, ACLNN_ERR_PARAM_NULLPTR, "mixed must not be nullptr.");
    if (useCombine) {
        CHECK_COND(injectOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR, "inject is required when w_gate is provided.");
        CHECK_COND(injectOptional->GetDataType() == x->GetDataType() &&
                       injectOptional->GetViewShape().GetDimNum() == HC_ACTIVATION_RANK &&
                       injectOptional->GetViewShape().GetDim(0) == x->GetViewShape().GetDim(0) &&
                       injectOptional->GetViewShape().GetDim(1) == x->GetViewShape().GetDim(1) &&
                       injectOptional->GetViewShape().GetDim(HC_FEATURE_DIM) == channels,
                   ACLNN_ERR_PARAM_INVALID, "inject must match x dtype and shape [B,S,C].");
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    auto xCont = l0op::Contiguous(x, executorPtr);
    auto wDownCont = l0op::Contiguous(wDown, executorPtr);
    auto wUpCont = l0op::Contiguous(wUp, executorPtr);
    CHECK_RET(xCont != nullptr && wDownCont != nullptr && wUpCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* wGateCont = nullptr;
    if (wGateOptional != nullptr) {
        wGateCont = l0op::Contiguous(wGateOptional, executorPtr);
        CHECK_RET(wGateCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* gammaCont = nullptr;
    if (gammaOptional != nullptr) {
        gammaCont = l0op::Contiguous(gammaOptional, executorPtr);
        CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outputs = l0op::TextGatedResidual(xCont, wDownCont, wUpCont, wGateCont, gammaCont, channels,
                                           static_cast<float>(eps), needBackward, executorPtr);
    CHECK_RET(outputs[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[0], mixed, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (useCombine) {
        CHECK_RET(l0op::ViewCopy(outputs[1], injectOptional, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (needBackward) {
        CHECK_RET(l0op::ViewCopy(outputs[2], invRmsOptional, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(outputs[3], downPreActOptional, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(outputs[4], upPreActOptional, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTextGatedResidual(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTextGatedResidual);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "TextGatedResidual launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
