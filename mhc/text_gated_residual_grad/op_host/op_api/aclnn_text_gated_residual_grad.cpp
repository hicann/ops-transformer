/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_text_gated_residual_grad.h"
#include "text_gated_residual_grad.h"

#include <limits>

#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t INPUT_RANK = 3;
constexpr int64_t MATRIX_RANK = 2;
constexpr int64_t BATCH_DIM = 0;
constexpr int64_t SEQUENCE_DIM = 1;
constexpr int64_t FEATURE_DIM = 2;
constexpr int64_t MATRIX_ROW_DIM = 0;
constexpr size_t OUT_DX = 0;
constexpr size_t OUT_DW_DOWN = 1;
constexpr size_t OUT_DW_UP = 2;
constexpr size_t OUT_DW_GATE = 3;
constexpr size_t OUT_DGAMMA = 4;

struct TextGatedResidualGradParams {
    const aclTensor* gradMixed = nullptr;
    const aclTensor* x = nullptr;
    const aclTensor* wDown = nullptr;
    const aclTensor* wUp = nullptr;
    const aclTensor* invRms = nullptr;
    const aclTensor* downPreAct = nullptr;
    const aclTensor* up = nullptr;
    const aclTensor* gradInject = nullptr;
    const aclTensor* wGate = nullptr;
    const aclTensor* gamma = nullptr;
    const aclTensor* inject = nullptr;
    int64_t channels = 0;
    aclTensor* dx = nullptr;
    aclTensor* dwDown = nullptr;
    aclTensor* dwUp = nullptr;
    aclTensor* dwGate = nullptr;
    aclTensor* dgamma = nullptr;
};

aclnnStatus CheckNotNull(const TextGatedResidualGradParams& p)
{
    CHECK_COND(p.gradMixed != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gradMixed must not be nullptr.");
    CHECK_COND(p.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(p.wDown != nullptr, ACLNN_ERR_PARAM_NULLPTR, "wDown must not be nullptr.");
    CHECK_COND(p.wUp != nullptr, ACLNN_ERR_PARAM_NULLPTR, "wUp must not be nullptr.");
    CHECK_COND(p.invRms != nullptr, ACLNN_ERR_PARAM_NULLPTR, "invRms must not be nullptr.");
    CHECK_COND(p.downPreAct != nullptr, ACLNN_ERR_PARAM_NULLPTR, "downPreAct must not be nullptr.");
    CHECK_COND(p.up != nullptr, ACLNN_ERR_PARAM_NULLPTR, "up must not be nullptr.");
    CHECK_COND(p.dx != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dx must not be nullptr.");
    CHECK_COND(p.dwDown != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dwDown must not be nullptr.");
    CHECK_COND(p.dwUp != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dwUp must not be nullptr.");
    CHECK_COND(p.wGate != nullptr || (p.gradInject == nullptr && p.inject == nullptr && p.dwGate == nullptr),
               ACLNN_ERR_PARAM_NULLPTR, "wGate is required when gradInject, inject or dwGate is given.");
    CHECK_COND(p.gamma != nullptr || p.dgamma == nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "gamma is required when dgamma is given.");
    if (p.wGate != nullptr) {
        CHECK_COND(p.gradInject != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "gradInject must not be nullptr when wGate is given.");
        CHECK_COND(p.inject != nullptr, ACLNN_ERR_PARAM_NULLPTR, "inject must not be nullptr when wGate is given.");
        CHECK_COND(p.dwGate != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dwGate must not be nullptr when wGate is given.");
    }
    if (p.gamma != nullptr) {
        CHECK_COND(p.dgamma != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dgamma must not be nullptr when gamma is given.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const TextGatedResidualGradParams& p)
{
    const std::initializer_list<op::DataType> supportDtype = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
    OP_CHECK_DTYPE_NOT_SUPPORT(p.x, supportDtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.gradMixed, supportDtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.wDown, supportDtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.wUp, supportDtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.downPreAct, supportDtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(p.x, p.gradMixed, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(p.x, p.wDown, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(p.x, p.wUp, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(p.x, p.downPreAct, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(p.x, p.up, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(p.x, p.dx, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(p.invRms, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(p.wDown, p.dwDown, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(p.wUp, p.dwUp, return ACLNN_ERR_PARAM_INVALID);
    if (p.wGate != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(p.x, p.gradInject, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SAME(p.x, p.wGate, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SAME(p.x, p.inject, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SAME(p.wGate, p.dwGate, return ACLNN_ERR_PARAM_INVALID);
    }
    if (p.gamma != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(p.gamma, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_MATCH(p.dgamma, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorShape(const aclTensor* t, std::initializer_list<int64_t> expect, const char* name)
{
    auto shape = t->GetViewShape();
    CHECK_COND(shape.GetDimNum() == static_cast<int64_t>(expect.size()), ACLNN_ERR_PARAM_INVALID,
               "%s dim num mismatch, expect %zu but got %zu", name, expect.size(), shape.GetDimNum());
    size_t i = 0;
    for (auto e : expect) {
        CHECK_COND(shape.GetDim(i) == e, ACLNN_ERR_PARAM_INVALID, "%s dim[%zu] expect %ld but got %ld", name, i, e,
                   shape.GetDim(i));
        i++;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShape(const TextGatedResidualGradParams& p)
{
    auto xShape = p.x->GetViewShape();
    CHECK_COND(xShape.GetDimNum() == INPUT_RANK, ACLNN_ERR_PARAM_INVALID, "x dim num should be 3, but got %zu",
               xShape.GetDimNum());
    int64_t b = xShape.GetDim(BATCH_DIM);
    int64_t s = xShape.GetDim(SEQUENCE_DIM);
    int64_t d = xShape.GetDim(FEATURE_DIM);
    CHECK_COND(b >= 0 && s >= 0, ACLNN_ERR_PARAM_INVALID, "B/S must be non-negative, but got B=%ld, S=%ld", b, s);
    CHECK_COND(s == 0 || b <= std::numeric_limits<int64_t>::max() / s, ACLNN_ERR_PARAM_INVALID,
               "B*S overflows int64, got B=%ld, S=%ld", b, s);
    CHECK_COND(d > 0, ACLNN_ERR_PARAM_INVALID, "x last dim must be positive, but got %ld", d);
    CHECK_COND(p.channels > 0 && d % p.channels == 0, ACLNN_ERR_PARAM_INVALID,
               "channels %ld invalid for x last dim %ld", p.channels, d);
    int64_t h = d / p.channels;
    CHECK_COND(p.wDown->GetViewShape().GetDimNum() == MATRIX_RANK, ACLNN_ERR_PARAM_INVALID,
               "wDown dim num should be 2, but got %zu", p.wDown->GetViewShape().GetDimNum());
    int64_t r = p.wDown->GetViewShape().GetDim(MATRIX_ROW_DIM);
    CHECK_COND(h > 0 && r > 0, ACLNN_ERR_PARAM_INVALID, "invalid channels/H or R.");

    CHECK_RET(CheckTensorShape(p.gradMixed, {b, s, h}, "gradMixed") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.x, {b, s, d}, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.wDown, {r, d}, "wDown") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.wUp, {d, r}, "wUp") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.invRms, {b, s, p.channels}, "invRms") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.downPreAct, {b, s, r}, "downPreAct") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.up, {b, s, d}, "up") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.dx, {b, s, d}, "dx") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.dwDown, {r, d}, "dwDown") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(p.dwUp, {d, r}, "dwUp") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (p.wGate != nullptr) {
        CHECK_RET(CheckTensorShape(p.wGate, {p.channels, d}, "wGate") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorShape(p.gradInject, {b, s, p.channels}, "gradInject") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorShape(p.inject, {b, s, p.channels}, "inject") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorShape(p.dwGate, {p.channels, d}, "dwGate") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    if (p.gamma != nullptr) {
        CHECK_RET(CheckTensorShape(p.gamma, {d}, "gamma") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorShape(p.dgamma, {d}, "dgamma") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const TextGatedResidualGradParams& p)
{
    CHECK_COND(CheckNotNull(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR, "required param is nullptr.");
    CHECK_COND(CheckDtype(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dtype.");
    CHECK_COND(CheckShape(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape.");
    return ACLNN_SUCCESS;
}

const aclTensor* ContiguousOrNull(const aclTensor* t, aclOpExecutor* executor)
{
    if (t == nullptr) {
        return nullptr;
    }
    return l0op::Contiguous(t, executor);
}

aclnnStatus MakeContiguousInputs(const TextGatedResidualGradParams& p, TextGatedResidualGradParams& contiguous,
                                 aclOpExecutor* executor)
{
    // Make inputs contiguous.
    contiguous.gradMixed = l0op::Contiguous(p.gradMixed, executor);
    CHECK_RET(contiguous.gradMixed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguous.gradInject = ContiguousOrNull(p.gradInject, executor);
    contiguous.x = l0op::Contiguous(p.x, executor);
    CHECK_RET(contiguous.x != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguous.wDown = l0op::Contiguous(p.wDown, executor);
    CHECK_RET(contiguous.wDown != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguous.wUp = l0op::Contiguous(p.wUp, executor);
    CHECK_RET(contiguous.wUp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguous.wGate = ContiguousOrNull(p.wGate, executor);
    contiguous.gamma = ContiguousOrNull(p.gamma, executor);
    CHECK_RET(p.wGate == nullptr || contiguous.wGate != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(p.gamma == nullptr || contiguous.gamma != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(p.gradInject == nullptr || contiguous.gradInject != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguous.invRms = l0op::Contiguous(p.invRms, executor);
    CHECK_RET(contiguous.invRms != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguous.inject = ContiguousOrNull(p.inject, executor);
    CHECK_RET(p.inject == nullptr || contiguous.inject != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguous.downPreAct = l0op::Contiguous(p.downPreAct, executor);
    CHECK_RET(contiguous.downPreAct != nullptr, ACLNN_ERR_INNER_NULLPTR);
    contiguous.up = l0op::Contiguous(p.up, executor);
    CHECK_RET(contiguous.up != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnTextGatedResidualGradGetWorkspaceSize(
    const aclTensor* gradMixed, const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp,
    const aclTensor* invRms, const aclTensor* downPreAct, const aclTensor* up, const aclTensor* gradInjectOptional,
    const aclTensor* wGateOptional, const aclTensor* gammaOptional, const aclTensor* injectOptional, int64_t channels,
    aclTensor* dx, aclTensor* dwDown, aclTensor* dwUp, aclTensor* dwGateOptional, aclTensor* dgammaOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnTextGatedResidualGrad,
                   DFX_IN(gradMixed, x, wDown, wUp, invRms, downPreAct, up, gradInjectOptional, wGateOptional,
                          gammaOptional, injectOptional),
                   DFX_OUT(dx, dwDown, dwUp, dwGateOptional, dgammaOptional));

    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");

    TextGatedResidualGradParams params{gradMixed,
                                       x,
                                       wDown,
                                       wUp,
                                       invRms,
                                       downPreAct,
                                       up,
                                       gradInjectOptional,
                                       wGateOptional,
                                       gammaOptional,
                                       injectOptional,
                                       channels,
                                       dx,
                                       dwDown,
                                       dwUp,
                                       dwGateOptional,
                                       dgammaOptional};
    aclnnStatus ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto contiguous = params;
    ret = MakeContiguousInputs(params, contiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto result = l0op::TextGatedResidualGrad(contiguous.gradMixed, contiguous.x, contiguous.wDown, contiguous.wUp,
                                              contiguous.invRms, contiguous.downPreAct, contiguous.up,
                                              contiguous.gradInject, contiguous.wGate, contiguous.gamma,
                                              contiguous.inject, channels, uniqueExecutor.get());
    CHECK_RET(std::get<OUT_DX>(result) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyIfNotNull = [&uniqueExecutor](aclTensor* out, aclTensor* userOut) -> aclnnStatus {
        if (userOut == nullptr || userOut->IsEmpty()) {
            return ACLNN_SUCCESS;
        }
        auto copiedOutput = l0op::ViewCopy(out, userOut, uniqueExecutor.get());
        CHECK_RET(copiedOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };
    CHECK_RET(viewCopyIfNotNull(std::get<OUT_DX>(result), dx) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(viewCopyIfNotNull(std::get<OUT_DW_DOWN>(result), dwDown) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(viewCopyIfNotNull(std::get<OUT_DW_UP>(result), dwUp) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(viewCopyIfNotNull(std::get<OUT_DW_GATE>(result), dwGateOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    if (gammaOptional != nullptr && dgammaOptional != nullptr) {
        CHECK_RET(viewCopyIfNotNull(std::get<OUT_DGAMMA>(result), dgammaOptional) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTextGatedResidualGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTextGatedResidualGrad);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in TextGatedResidualGrad launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
