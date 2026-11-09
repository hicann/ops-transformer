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

#include "log/log.h"
#include "register/op_impl_registry.h"

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
        OP_LOGE(context->GetNodeName(), "%s rank should be %zu, but got %zu.", name, rank, shape->GetDimNum());
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus CheckDim(const gert::Shape* shape, size_t idx, int64_t expect, const char* name,
                     gert::InferShapeContext* context)
{
    const int64_t actual = shape->GetDim(idx);
    if (!DimEqualOrUnknown(actual, expect)) {
        OP_LOGE(context->GetNodeName(), "%s dim[%zu] should be %ld, but got %ld.", name, idx, expect, actual);
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

        OP_CHECK_NULL_WITH_CONTEXT(context, gradMixedShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, wDownShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, wUpShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, invRmsShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, downPreShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, upShape);

        dxShape = GetOutputShapeByIrIndex(context, OUT_DX);
        dwDownShape = GetOutputShapeByIrIndex(context, OUT_DW_DOWN);
        dwUpShape = GetOutputShapeByIrIndex(context, OUT_DW_UP);
        dwGateShape = GetOutputShapeByIrIndex(context, OUT_DW_GATE);
        dgammaShape = GetOutputShapeByIrIndex(context, OUT_DGAMMA);
        OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, dwDownShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, dwUpShape);

        return GRAPH_SUCCESS;
    }

    graphStatus CheckAttributes()
    {
        const auto* attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const int64_t* channelsPtr = attrs->GetAttrPointer<int64_t>(CHANNELS_ATTR_INDEX);
        if (channelsPtr == nullptr) {
            OP_LOGE(context->GetNodeName(), "channels attr is null.");
            return GRAPH_FAILED;
        }
        c = *channelsPtr;
        if (c <= 0) {
            OP_LOGE(context->GetNodeName(), "channels should be positive, but got %ld.", c);
            return GRAPH_FAILED;
        }
        hasGate = !IsEmptyShape(wGateShape);
        if (hasGate && dwGateShape == nullptr) {
            OP_LOGE(context->GetNodeName(), "dw_gate output is required when w_gate is present.");
            return GRAPH_FAILED;
        }
        if (hasGate && (gradInjectShape == nullptr || injectShape == nullptr)) {
            OP_LOGE(context->GetNodeName(), "grad_inject and inject are required when w_gate is present.");
            return GRAPH_FAILED;
        }

        hasGamma = !IsEmptyShape(gammaShape);
        if (hasGamma && dgammaShape == nullptr) {
            OP_LOGE(context->GetNodeName(), "dgamma output is required when gamma is present.");
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
            OP_LOGE(context->GetNodeName(), "x last dim %ld is not divisible by channels %ld.", d, c);
            return GRAPH_FAILED;
        }
        h = (d == UNKNOWN_DIM) ? UNKNOWN_DIM : d / c;
        r = wDownShape->GetDim(FIRST_DIM);
        if (r != UNKNOWN_DIM && r <= 0) {
            OP_LOGE(context->GetNodeName(), "R must be positive or unknown, but got %ld.", r);
            return GRAPH_FAILED;
        }
        b = xShape->GetDim(FIRST_DIM);
        s = xShape->GetDim(SECOND_DIM);
        if ((b < 0 && b != UNKNOWN_DIM) || (s < 0 && s != UNKNOWN_DIM)) {
            OP_LOGE(context->GetNodeName(), "B/S must be non-negative or unknown, but got B=%ld, S=%ld.", b, s);
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

graphStatus SetOutputDtype(gert::InferDataTypeContext* context, size_t irIndex, ge::DataType dtype)
{
    const auto* instance = context->GetIrOutputInstanceInfo(irIndex);
    if (instance == nullptr || instance->GetInstanceNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    return context->SetOutputDataType(instance->GetInstanceStart(), dtype);
}

ge::graphStatus InferDataTypeForTextGatedResidualGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType xDtype = context->GetRequiredInputDataType(IDX_X);
    if (xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16) {
        OP_LOGE(context->GetNodeName(), "x dtype only supports bf16/fp16, but got %d.", static_cast<int32_t>(xDtype));
        return GRAPH_FAILED;
    }
    const size_t requiredSameDtypeInputs[] = {IDX_GRAD_MIXED, IDX_W_DOWN, IDX_W_UP, IDX_DOWN_PRE, IDX_UP};
    for (size_t idx : requiredSameDtypeInputs) {
        if (context->GetRequiredInputDataType(idx) != xDtype) {
            OP_LOGE(context->GetNodeName(), "input %zu dtype should match x.", idx);
            return GRAPH_FAILED;
        }
    }
    const size_t optionalSameDtypeInputs[] = {IDX_GRAD_INJECT, IDX_W_GATE, IDX_INJECT};
    for (size_t idx : optionalSameDtypeInputs) {
        const ge::DataType dtype = context->GetOptionalInputDataType(idx);
        if (dtype != ge::DT_UNDEFINED && dtype != xDtype) {
            OP_LOGE(context->GetNodeName(), "input %zu dtype should match x.", idx);
            return GRAPH_FAILED;
        }
    }
    const ge::DataType gammaDtype = context->GetOptionalInputDataType(IDX_GAMMA);
    if (gammaDtype != ge::DT_UNDEFINED && gammaDtype != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "gamma dtype only supports float32.");
        return GRAPH_FAILED;
    }
    if (context->GetRequiredInputDataType(IDX_INV_RMS) != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "inv_rms dtype only supports float32.");
        return GRAPH_FAILED;
    }

    if (SetOutputDtype(context, OUT_DX, xDtype) != ge::GRAPH_SUCCESS ||
        SetOutputDtype(context, OUT_DW_DOWN, xDtype) != ge::GRAPH_SUCCESS ||
        SetOutputDtype(context, OUT_DW_UP, xDtype) != ge::GRAPH_SUCCESS ||
        SetOutputDtype(context, OUT_DW_GATE, xDtype) != ge::GRAPH_SUCCESS ||
        SetOutputDtype(context, OUT_DGAMMA, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "failed to set output dtype.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(TextGatedResidualGrad)
    .InferShape(InferShapeForTextGatedResidualGrad)
    .InferDataType(InferDataTypeForTextGatedResidualGrad);

} // namespace ops
