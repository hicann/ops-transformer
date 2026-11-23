/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/text_gated_residual_grad_constants.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t INPUT_RANK = 3;
constexpr size_t MATRIX_RANK = 2;
constexpr size_t VECTOR_RANK = 1;
constexpr size_t FIRST_DIM = 0;
constexpr size_t SECOND_DIM = 1;
constexpr size_t FEATURE_DIM = 2;
constexpr size_t CHANNELS_ATTR_INDEX = 0;
constexpr size_t IDX_GRAD_MIXED = 0;
constexpr size_t IDX_X = 1;
constexpr size_t IDX_W_DOWN = 2;
constexpr size_t IDX_W_UP = 3;
constexpr size_t IDX_INV_RMS = 4;
constexpr size_t IDX_DOWN_PRE = 5;
constexpr size_t IDX_UP = 6;
constexpr size_t IDX_GRAD_INJECT = 7;
constexpr size_t IDX_W_GATE = 8;
constexpr size_t IDX_GAMMA = 9;
constexpr size_t IDX_INJECT = 10;

constexpr size_t OUT_DX = 0;
constexpr size_t OUT_DW_DOWN = 1;
constexpr size_t OUT_DW_UP = 2;
constexpr size_t OUT_DW_GATE = 3;
constexpr size_t OUT_DGAMMA = 4;

constexpr int64_t UNKNOWN_DIM = -1LL;
constexpr int64_t UNKNOWN_RANK = -2LL;

bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == VECTOR_RANK && shape->GetDim(FIRST_DIM) == UNKNOWN_RANK;
}

bool IsEmptyShape(const gert::Shape* shape)
{
    return shape == nullptr || (shape->GetDimNum() == VECTOR_RANK && shape->GetDim(FIRST_DIM) == 0);
}

bool DimEqualOrUnknown(int64_t actual, int64_t expect)
{
    // Only known, unequal dimensions are incompatible; an unknown dimension remains available for dynamic inference.
    return actual == expect || actual == UNKNOWN_DIM || expect == UNKNOWN_DIM;
}

gert::Shape* GetOutputShapeByIrIndex(gert::InferShapeContext* context, size_t irIndex)
{
    const auto* instance = context->GetIrOutputInstanceInfo(irIndex);
    if (instance == nullptr || instance->GetInstanceNum() == 0) {
        return nullptr;
    }
    return context->GetOutputShape(instance->GetInstanceStart());
}

void SetUnknownRank(gert::Shape* shape)
{
    if (shape == nullptr) {
        return;
    }
    shape->SetDimNum(0);
    shape->AppendDim(UNKNOWN_RANK);
}

void CopyShape(const gert::Shape* src, gert::Shape* dst)
{
    if (src == nullptr || dst == nullptr) {
        return;
    }
    dst->SetDimNum(src->GetDimNum());
    for (size_t i = 0; i < src->GetDimNum(); ++i) {
        dst->SetDim(i, src->GetDim(i));
    }
}

graphStatus CheckRank(const gert::Shape* shape, size_t rank, const char* name, gert::InferShapeContext* context)
{
    if (shape->GetDimNum() != rank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(Ops::Base::GetOpInfo(context), name, std::to_string(shape->GetDimNum()),
                                     std::to_string(rank));
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus CheckDim(const gert::Shape* shape, size_t idx, int64_t expect, const char* name,
                     gert::InferShapeContext* context)
{
    const int64_t actual = shape->GetDim(idx);
    if (!DimEqualOrUnknown(actual, expect)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Ops::Base::GetOpInfo(context), name, Ops::Base::ToString(*shape),
                                              "dim[" + std::to_string(idx) + "] must be " + std::to_string(expect));
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus CheckShape3(const gert::Shape* shape, int64_t d0, int64_t d1, int64_t d2, const char* name,
                        gert::InferShapeContext* context)
{
    if (CheckRank(shape, INPUT_RANK, name, context) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (CheckDim(shape, FIRST_DIM, d0, name, context) != GRAPH_SUCCESS ||
        CheckDim(shape, SECOND_DIM, d1, name, context) != GRAPH_SUCCESS ||
        CheckDim(shape, FEATURE_DIM, d2, name, context) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus CheckShape2(const gert::Shape* shape, int64_t d0, int64_t d1, const char* name,
                        gert::InferShapeContext* context)
{
    if (CheckRank(shape, MATRIX_RANK, name, context) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (CheckDim(shape, FIRST_DIM, d0, name, context) != GRAPH_SUCCESS ||
        CheckDim(shape, SECOND_DIM, d1, name, context) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

class TextGatedResidualGradShapeInference {
public:
    explicit TextGatedResidualGradShapeInference(gert::InferShapeContext* ctx)
        : context(ctx)
    {}

    graphStatus Infer()
    {
        if (LoadShapes() != GRAPH_SUCCESS || CheckAttributes() != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        if (PropagateUnknownRank()) {
            return GRAPH_SUCCESS;
        }
        if (ReadDimensions() != GRAPH_SUCCESS || CheckKnownShapes() != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        return SetOutputShapes();
    }

private:
    graphStatus LoadShapes()
    {
        gradMixedShape = context->GetRequiredInputShape(IDX_GRAD_MIXED);
        xShape = context->GetRequiredInputShape(IDX_X);
        wDownShape = context->GetRequiredInputShape(IDX_W_DOWN);
        wUpShape = context->GetRequiredInputShape(IDX_W_UP);
        invRmsShape = context->GetRequiredInputShape(IDX_INV_RMS);
        downPreShape = context->GetRequiredInputShape(IDX_DOWN_PRE);
        upShape = context->GetRequiredInputShape(IDX_UP);
        gradInjectShape = context->GetOptionalInputShape(IDX_GRAD_INJECT);
        wGateShape = context->GetOptionalInputShape(IDX_W_GATE);
        gammaShape = context->GetOptionalInputShape(IDX_GAMMA);
        injectShape = context->GetOptionalInputShape(IDX_INJECT);

        const std::pair<const gert::Shape*, const char*> requiredInputs[] = {
            {gradMixedShape, "grad_mixed"},
            {xShape, "x"},
            {wDownShape, "w_down"},
            {wUpShape, "w_up"},
            {invRmsShape, "inv_rms"},
            {downPreShape, "down_pre_act"},
            {upShape, "up"},
        };
        for (const auto& [shape, name] : requiredInputs) {
            if (shape == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT(Ops::Base::GetOpInfo(context), name);
                return GRAPH_FAILED;
            }
        }

        dxShape = GetOutputShapeByIrIndex(context, OUT_DX);
        dwDownShape = GetOutputShapeByIrIndex(context, OUT_DW_DOWN);
        dwUpShape = GetOutputShapeByIrIndex(context, OUT_DW_UP);
        dwGateShape = GetOutputShapeByIrIndex(context, OUT_DW_GATE);
        dgammaShape = GetOutputShapeByIrIndex(context, OUT_DGAMMA);
        const std::pair<gert::Shape*, const char*> requiredOutputs[] = {
            {dxShape, "dx"},
            {dwDownShape, "dw_down"},
            {dwUpShape, "dw_up"},
        };
        for (const auto& [shape, name] : requiredOutputs) {
            if (shape == nullptr) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context), name,
                                                         "required output shape is missing");
                return GRAPH_FAILED;
            }
        }

        return GRAPH_SUCCESS;
    }

    graphStatus CheckAttributes()
    {
        const auto* attrs = context->GetAttrs();
        const int64_t* channelsPtr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(CHANNELS_ATTR_INDEX);
        if (channelsPtr == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context), "channels",
                                                     "the required integer attribute is missing");
            return GRAPH_FAILED;
        }
        c = *channelsPtr;
        if (c <= 0 || c > TextGatedResidualGrad::HC_GRAD_MAX_CHANNELS) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                Ops::Base::GetOpInfo(context), "channels", std::to_string(c),
                "must be in [1, " + std::to_string(TextGatedResidualGrad::HC_GRAD_MAX_CHANNELS) + "]");
            return GRAPH_FAILED;
        }
        hasGate = !IsEmptyShape(wGateShape);
        if (hasGate && dwGateShape == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context), "dw_gate",
                                                     "the output is required when w_gate is provided");
            return GRAPH_FAILED;
        }
        if (hasGate && (gradInjectShape == nullptr || injectShape == nullptr)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context),
                                                     gradInjectShape == nullptr ? "grad_inject" : "inject",
                                                     "the input is required when w_gate is provided");
            return GRAPH_FAILED;
        }

        hasGamma = !IsEmptyShape(gammaShape);
        if (hasGamma && dgammaShape == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context), "dgamma",
                                                     "the output is required when gamma is provided");
            return GRAPH_FAILED;
        }

        return GRAPH_SUCCESS;
    }

    bool PropagateUnknownRank()
    {
        const gert::Shape* allInputs[] = {gradMixedShape, gradInjectShape, xShape,     wDownShape,
                                          wUpShape,       wGateShape,      gammaShape, invRmsShape,
                                          injectShape,    downPreShape,    upShape};
        for (const auto* shape : allInputs) {
            if (shape != nullptr && IsUnknownRank(shape)) {
                SetUnknownRank(dxShape);
                SetUnknownRank(dwDownShape);
                SetUnknownRank(dwUpShape);
                SetUnknownRank(dwGateShape);
                if (dgammaShape != nullptr) {
                    SetUnknownRank(dgammaShape);
                }
                return true;
            }
        }

        return false;
    }

    graphStatus ReadDimensions()
    {
        if (CheckRank(xShape, INPUT_RANK, "x", context) != GRAPH_SUCCESS ||
            CheckRank(wDownShape, MATRIX_RANK, "w_down", context) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        d = xShape->GetDim(FEATURE_DIM);
        if (d != UNKNOWN_DIM && (d <= 0 || d % c != 0)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                Ops::Base::GetOpInfo(context), "x", Ops::Base::ToString(*xShape),
                "the last dimension must be positive and divisible by channels=" + std::to_string(c));
            return GRAPH_FAILED;
        }
        h = (d == UNKNOWN_DIM) ? UNKNOWN_DIM : d / c;
        if (h != UNKNOWN_DIM && h % TextGatedResidualGrad::HC_GRAD_VECTOR_ALIGN != 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Ops::Base::GetOpInfo(context), "x", Ops::Base::ToString(*xShape),
                                                  "the last dimension divided by channels=" + std::to_string(c) +
                                                      " gives H=" + std::to_string(h) +
                                                      ", which must be a multiple of 16");
            return GRAPH_FAILED;
        }
        r = wDownShape->GetDim(FIRST_DIM);
        if (r != UNKNOWN_DIM && (r <= 0 || r > TextGatedResidualGrad::HC_GRAD_MAX_PROJECTION ||
                                 r % TextGatedResidualGrad::HC_GRAD_GATE_ALIGN != 0)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                Ops::Base::GetOpInfo(context), "w_down", Ops::Base::ToString(*wDownShape),
                "the first dimension R must be a multiple of 16 in [1, " +
                    std::to_string(TextGatedResidualGrad::HC_GRAD_MAX_PROJECTION) + "]");
            return GRAPH_FAILED;
        }
        b = xShape->GetDim(FIRST_DIM);
        s = xShape->GetDim(SECOND_DIM);
        if ((b < 0 && b != UNKNOWN_DIM) || (s < 0 && s != UNKNOWN_DIM)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Ops::Base::GetOpInfo(context), "x", Ops::Base::ToString(*xShape),
                                                  "the batch and sequence dimensions must be non-negative or unknown");
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    graphStatus CheckKnownShapes()
    {
        if (CheckShape3(gradMixedShape, b, s, h, "grad_mixed", context) != GRAPH_SUCCESS ||
            CheckShape3(xShape, b, s, d, "x", context) != GRAPH_SUCCESS ||
            CheckShape2(wDownShape, r, d, "w_down", context) != GRAPH_SUCCESS ||
            CheckShape2(wUpShape, d, r, "w_up", context) != GRAPH_SUCCESS ||
            CheckShape3(invRmsShape, b, s, c, "inv_rms", context) != GRAPH_SUCCESS ||
            CheckShape3(downPreShape, b, s, r, "down_pre_act", context) != GRAPH_SUCCESS ||
            CheckShape3(upShape, b, s, d, "up", context) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }

        if (hasGate && (CheckShape2(wGateShape, c, d, "w_gate", context) != GRAPH_SUCCESS ||
                        CheckShape3(gradInjectShape, b, s, c, "grad_inject", context) != GRAPH_SUCCESS ||
                        CheckShape3(injectShape, b, s, c, "inject", context) != GRAPH_SUCCESS)) {
            return GRAPH_FAILED;
        }

        if (hasGamma && (CheckRank(gammaShape, VECTOR_RANK, "gamma", context) != GRAPH_SUCCESS ||
                         CheckDim(gammaShape, FIRST_DIM, d, "gamma", context) != GRAPH_SUCCESS)) {
            return GRAPH_FAILED;
        }

        return GRAPH_SUCCESS;
    }

    graphStatus SetOutputShapes()
    {
        CopyShape(xShape, dxShape);
        CopyShape(wDownShape, dwDownShape);
        CopyShape(wUpShape, dwUpShape);
        if (dwGateShape != nullptr) {
            if (hasGate) {
                CopyShape(wGateShape, dwGateShape);
            } else {
                dwGateShape->SetDimNum(1);
                dwGateShape->SetDim(0, 0);
            }
        }
        if (dgammaShape != nullptr) {
            if (hasGamma) {
                CopyShape(gammaShape, dgammaShape);
            } else {
                dgammaShape->SetDimNum(1);
                dgammaShape->SetDim(0, 0);
            }
        }
        return GRAPH_SUCCESS;
    }

    gert::InferShapeContext* context;
    const gert::Shape* gradMixedShape = nullptr;
    const gert::Shape* xShape = nullptr;
    const gert::Shape* wDownShape = nullptr;
    const gert::Shape* wUpShape = nullptr;
    const gert::Shape* invRmsShape = nullptr;
    const gert::Shape* downPreShape = nullptr;
    const gert::Shape* upShape = nullptr;
    const gert::Shape* gradInjectShape = nullptr;
    const gert::Shape* wGateShape = nullptr;
    const gert::Shape* gammaShape = nullptr;
    const gert::Shape* injectShape = nullptr;
    gert::Shape* dxShape = nullptr;
    gert::Shape* dwDownShape = nullptr;
    gert::Shape* dwUpShape = nullptr;
    gert::Shape* dwGateShape = nullptr;
    gert::Shape* dgammaShape = nullptr;
    int64_t b = 0;
    int64_t s = 0;
    int64_t c = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t r = 0;
    bool hasGate = false;
    bool hasGamma = false;
};

ge::graphStatus InferShapeForTextGatedResidualGrad(gert::InferShapeContext* context)
{
    return TextGatedResidualGradShapeInference(context).Infer();
}

graphStatus SetOutputDtype(gert::InferDataTypeContext* context, size_t irIndex, ge::DataType dtype, const char* name)
{
    const auto* instance = context->GetIrOutputInstanceInfo(irIndex);
    if (instance == nullptr || instance->GetInstanceNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    const auto status = context->SetOutputDataType(instance->GetInstanceStart(), dtype);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "SetOutputDataType failed for %s, dtype=%s, status=%u.", name,
                               Ops::Base::ToString(dtype).c_str(), static_cast<unsigned>(status));
    }
    return status;
}

ge::graphStatus InferDataTypeForTextGatedResidualGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType xDtype = context->GetRequiredInputDataType(IDX_X);
    if (xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE(Ops::Base::GetOpInfo(context), "x", Ops::Base::ToString(xDtype), "BF16 or FP16");
        return GRAPH_FAILED;
    }
    constexpr std::pair<size_t, const char*> requiredSameDtypeInputs[] = {
        {IDX_GRAD_MIXED, "grad_mixed"}, {IDX_W_DOWN, "w_down"}, {IDX_W_UP, "w_up"},
        {IDX_DOWN_PRE, "down_pre_act"}, {IDX_UP, "up"},
    };
    for (const auto& [idx, name] : requiredSameDtypeInputs) {
        const ge::DataType dtype = context->GetRequiredInputDataType(idx);
        if (dtype != xDtype) {
            OP_LOGE_FOR_INVALID_DTYPE(Ops::Base::GetOpInfo(context), name, Ops::Base::ToString(dtype),
                                      Ops::Base::ToString(xDtype));
            return GRAPH_FAILED;
        }
    }
    constexpr std::pair<size_t, const char*> optionalSameDtypeInputs[] = {
        {IDX_GRAD_INJECT, "grad_inject"},
        {IDX_W_GATE, "w_gate"},
        {IDX_INJECT, "inject"},
    };
    for (const auto& [idx, name] : optionalSameDtypeInputs) {
        const ge::DataType dtype = context->GetOptionalInputDataType(idx);
        if (dtype != ge::DT_UNDEFINED && dtype != xDtype) {
            OP_LOGE_FOR_INVALID_DTYPE(Ops::Base::GetOpInfo(context), name, Ops::Base::ToString(dtype),
                                      Ops::Base::ToString(xDtype));
            return GRAPH_FAILED;
        }
    }
    const ge::DataType gammaDtype = context->GetOptionalInputDataType(IDX_GAMMA);
    if (gammaDtype != ge::DT_UNDEFINED && gammaDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(Ops::Base::GetOpInfo(context), "gamma", Ops::Base::ToString(gammaDtype), "FP32");
        return GRAPH_FAILED;
    }
    const ge::DataType invRmsDtype = context->GetRequiredInputDataType(IDX_INV_RMS);
    if (invRmsDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(Ops::Base::GetOpInfo(context), "inv_rms", Ops::Base::ToString(invRmsDtype), "FP32");
        return GRAPH_FAILED;
    }

    if (SetOutputDtype(context, OUT_DX, xDtype, "dx") != ge::GRAPH_SUCCESS ||
        SetOutputDtype(context, OUT_DW_DOWN, xDtype, "dw_down") != ge::GRAPH_SUCCESS ||
        SetOutputDtype(context, OUT_DW_UP, xDtype, "dw_up") != ge::GRAPH_SUCCESS ||
        SetOutputDtype(context, OUT_DW_GATE, xDtype, "dw_gate") != ge::GRAPH_SUCCESS ||
        SetOutputDtype(context, OUT_DGAMMA, ge::DT_FLOAT, "dgamma") != ge::GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(TextGatedResidualGrad)
    .InferShape(InferShapeForTextGatedResidualGrad)
    .InferDataType(InferDataTypeForTextGatedResidualGrad);

} // namespace ops
