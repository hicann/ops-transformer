/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_finalize_routing_v5.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe_finalize_routing_v2.h"
#include "moe_finalize_routing_common.h"
#include "common/include/op_host/op_const_def_transformer.h"

using namespace op;

static aclnnStatus CheckMx4BufferShapes(const aclTensor* expandedX, const aclTensor* expandedRowIdx,
                                        const aclTensor* bias, const aclTensor* expandedXScale, int64_t mode, int64_t k)
{
    const auto& expandedShape = expandedX->GetViewShape();
    const auto h = expandedShape.GetDim(expandedShape.GetDimNum() - 1);
    const auto indexCount = expandedRowIdx->GetViewShape().GetDim(0);
    CHECK_RET(h > 0 && h % 2 == 0 && indexCount >= 0 && indexCount % k == 0, ACLNN_ERR_PARAM_INVALID);
    const auto n = indexCount / k;
    const auto rowsPerExpert = MoeFinalizeRoutingCheck::IsExpandedX3D(mode) ? expandedShape.GetDim(1) : 1;
    int64_t rows;
    CHECK_RET(expandedShape.GetDim(0) >= 0 && rowsPerExpert >= 0 &&
                  !__builtin_mul_overflow(expandedShape.GetDim(0), rowsPerExpert, &rows) && (n == 0 || rows > 0),
              ACLNN_ERR_PARAM_INVALID);
    constexpr int64_t blockSize = 64;
    constexpr int64_t scalePairSize = 2;
    const int64_t blocks = h / blockSize + (h % blockSize != 0);
    CHECK_RET((expandedXScale->GetViewShape() == op::Shape{rows, blocks, scalePairSize}), ACLNN_ERR_PARAM_INVALID);

    const auto biasRows = bias == nullptr ? 0 : bias->GetViewShape().GetDim(0);
    int64_t bytes;
    CHECK_RET((bias == nullptr || biasRows > 0) && !__builtin_mul_overflow(rows, h / 2, &bytes) &&
                  !__builtin_mul_overflow(rows, blocks * scalePairSize, &bytes) &&
                  !__builtin_mul_overflow(n, h, &bytes) && !__builtin_mul_overflow(bytes, sizeof(uint16_t), &bytes) &&
                  !__builtin_mul_overflow(indexCount, sizeof(int32_t), &bytes) &&
                  !__builtin_mul_overflow(biasRows, h, &bytes) &&
                  !__builtin_mul_overflow(bytes, sizeof(uint16_t), &bytes),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMx4(const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1,
                            const aclTensor* x2, const aclTensor* bias, const aclTensor* scales,
                            const aclTensor* expertIdx, const aclTensor* expandedXScale, const aclTensor* out,
                            int64_t mode, int64_t k)
{
    using namespace MoeFinalizeRoutingCheck;
    OP_CHECK_NULL(expandedXScale, return ACLNN_ERR_PARAM_NULLPTR);
    const auto arch = GetCurrentPlatformInfo().GetCurNpuArch();
    CHECK_RET(arch == Ops::Base::DAV_3510 || arch == Ops::Base::DAV_9201 || arch == Ops::Base::DAV_9202,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, expertIdx, out, true) &&
                  CheckFormatValid(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, expertIdx, nullptr, nullptr,
                                   nullptr, nullptr, out, mode, true),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(scales == nullptr ||
                  (scales->GetDataType() == DataType::DT_FLOAT && scales->GetStorageFormat() == Format::FORMAT_ND),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(op::IsContiguous(expandedX) && op::IsContiguous(expandedXScale) && op::IsContiguous(out),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(mode >= 0 && mode <= 3 && k > 0 && (x2 == nullptr || x1 != nullptr) &&
                  (bias == nullptr || expertIdx != nullptr),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeValid(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, nullptr, nullptr, nullptr,
                              nullptr, out, mode, k),
              ACLNN_ERR_PARAM_INVALID);

    return CheckMx4BufferShapes(expandedX, expandedRowIdx, bias, expandedXScale, mode, k);
}

static aclnnStatus CheckV5(const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1Optional,
                           const aclTensor* x2Optional, const aclTensor* biasOptional, const aclTensor* scalesOptional,
                           const aclTensor* expertIdxOptional, const aclTensor* xOptional,
                           const aclTensor* alpha1Optional, const aclTensor* alpha2Optional, const aclTensor* vOptional,
                           const aclTensor* expandedXScaleOptional, const aclTensor* out, int64_t dropPadMode,
                           int64_t k)
{
    CHECK_RET(MoeFinalizeRoutingCheck::CheckNotNull(expandedX, expandedRowIdx, out), ACLNN_ERR_PARAM_NULLPTR);
    if (expandedX->GetDataType() == DataType::DT_FLOAT4_E2M1) {
        return CheckMx4(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                        expertIdxOptional, expandedXScaleOptional, out, dropPadMode, k);
    }
    CHECK_RET(static_cast<ge::DataType>(expandedX->GetDataType()) != Ops::Transformer::DT_HIFLOAT4,
              ACLNN_ERR_PARAM_INVALID);
    if (MoeFinalizeRoutingCheck::Is310P()) {
        return MoeFinalizeRoutingCheck::CheckParams310P(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional,
                                                        scalesOptional, expertIdxOptional, out, dropPadMode);
    } else if (MoeFinalizeRoutingCheck::IsCommonValidationChip()) {
        auto ret = MoeFinalizeRoutingCheck::CheckParams(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional,
                                                        scalesOptional, expertIdxOptional, xOptional, alpha1Optional,
                                                        alpha2Optional, vOptional, out, dropPadMode);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        CHECK_RET(MoeFinalizeRoutingCheck::CheckConstExpertDtype(expandedX, xOptional, alpha1Optional, alpha2Optional,
                                                                 vOptional),
                  ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
    return ACLNN_SUCCESS;
}

struct V5Inputs {
    const aclTensor* expandedX;
    const aclTensor* expandedRowIdx;
    const aclTensor* x1;
    const aclTensor* x2;
    const aclTensor* bias;
    const aclTensor* scales;
    const aclTensor* expertIdx;
    const aclTensor* x;
    const aclTensor* alpha1;
    const aclTensor* alpha2;
    const aclTensor* v;
    const aclTensor* expandedXScale;
};

static aclnnStatus PrepareV5Inputs(V5Inputs& inputs, aclOpExecutor* executor)
{
    if (inputs.expandedX->GetDataType() != DataType::DT_FLOAT4_E2M1) {
        inputs.expandedX = l0op::Contiguous(inputs.expandedX, executor);
        CHECK_RET(inputs.expandedX != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    for (auto tensor : {&inputs.expandedRowIdx, &inputs.x1, &inputs.x2, &inputs.bias, &inputs.scales, &inputs.expertIdx,
                        &inputs.x, &inputs.alpha1, &inputs.alpha2, &inputs.v, &inputs.expandedXScale}) {
        if (*tensor != nullptr) {
            *tensor = l0op::Contiguous(*tensor, executor);
            CHECK_RET(*tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV5GetWorkspaceSize(
    const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1Optional,
    const aclTensor* x2Optional, const aclTensor* biasOptional, const aclTensor* scalesOptional,
    const aclTensor* expertIdxOptional, const aclTensor* xOptional, const aclTensor* alpha1Optional,
    const aclTensor* alpha2Optional, const aclTensor* vOptional, const aclTensor* expandedXScaleOptional,
    int64_t dropPadMode, const aclIntArray* zeroExpertRange, const aclIntArray* copyExpertRange,
    const aclIntArray* constantExpertRange, int64_t k, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    const bool isMx4 = expandedX != nullptr && expandedX->GetDataType() == DataType::DT_FLOAT4_E2M1;
    const aclTensor* dequantScale = isMx4 ? expandedXScaleOptional : nullptr;
    L2_DFX_PHASE_1(aclnnMoeFinalizeRoutingV5,
                   DFX_IN(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                          expertIdxOptional, xOptional, alpha1Optional, alpha2Optional, vOptional, dequantScale,
                          dropPadMode, zeroExpertRange, copyExpertRange, constantExpertRange, k),
                   DFX_OUT(out));

    const auto isEmptyRange = [](const aclIntArray* range) { return range == nullptr || range->Size() == 0; };
    OP_CHECK(
        !isMx4 ||
            (xOptional == nullptr && alpha1Optional == nullptr && alpha2Optional == nullptr && vOptional == nullptr &&
             isEmptyRange(zeroExpertRange) && isEmptyRange(copyExpertRange) && isEmptyRange(constantExpertRange)),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            "aclnnMoeFinalizeRoutingV5", "special expert tensors and ranges",
            "MXFP4 does not support zero, copy or constant experts. Special expert tensors and ranges must be empty."),
        return ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret =
        CheckV5(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional, expertIdxOptional,
                xOptional, alpha1Optional, alpha2Optional, vOptional, dequantScale, out, dropPadMode, k);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    V5Inputs inputs{expandedX,         expandedRowIdx, x1Optional,     x2Optional,     biasOptional, scalesOptional,
                    expertIdxOptional, xOptional,      alpha1Optional, alpha2Optional, vOptional,    dequantScale};
    ret = PrepareV5Inputs(inputs, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto out_ = l0op::MoeFinalizeRoutingV2(inputs.expandedX, inputs.expandedRowIdx, inputs.x1, inputs.x2, inputs.bias,
                                           inputs.scales, inputs.expertIdx, inputs.x, inputs.alpha1, inputs.alpha2,
                                           inputs.v, inputs.expandedXScale, dropPadMode, zeroExpertRange,
                                           copyExpertRange, constantExpertRange, k, out, uniqueExecutor.get());
    CHECK_RET(out_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyOutResult = l0op::ViewCopy(out_, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV5(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeFinalizeRoutingV5);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
