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
 * \file norm_rope_concat_grad_infershape.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "norm_rope_concat_grad_base_host.h"

using namespace ge;
using namespace NormRopeConcatGrad;
namespace ops {
static ge::graphStatus CheckShape(gert::InferShapeContext *context, const gert::Shape *shape, int64_t batch, int64_t head,
                                  int64_t dim, int64_t &seq)
{
    if (shape->GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(context, "Input must be 4D tensors(B, S, H, D).");
        return ge::GRAPH_FAILED;
    }
    if (shape->GetDim(BATCH_DIM) != batch || shape->GetDim(HEAD_DIM) != head || shape->GetDim(DIM_DIM) != dim) {
        OP_LOGE(context, "Input must have the same batch, head and dim as Query.");
        return ge::GRAPH_FAILED;
    }
    seq = shape->GetDim(SEQ_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTransposeShape(gert::InferShapeContext *context, const gert::Shape *shape, int64_t batch,
                                           int64_t head, int64_t dim, int64_t &seq)
{
    if (shape->GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(context, "Input must be 4D tensors(B, S, H, D).");
        return ge::GRAPH_FAILED;
    }
    if (shape->GetDim(BATCH_DIM) != batch || shape->GetDim(T_HEAD_DIM) != head || shape->GetDim(DIM_DIM) != dim) {
        OP_LOGE(context, "Input must have the same batch, head and dim as Query.");
        return ge::GRAPH_FAILED;
    }
    seq = shape->GetDim(T_SEQ_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4NormRopeConcatGrad(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("InferShape4NormRopeConcatGrad", "context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Enter InferShape4NormRopeConcatGrad.");

    const gert::Shape *gradQueryOutputShape =
        context->GetInputShape(static_cast<size_t>(InputIndexBackward::GRAD_QUERY_OUTPUT_INDEX));
    const gert::Shape *gradKeyOutputShape =
        context->GetInputShape(static_cast<size_t>(InputIndexBackward::GRAD_KEY_OUTPUT_INDEX));
    const gert::Shape *gradValueOutputShape =
        context->GetInputShape(static_cast<size_t>(InputIndexBackward::GRAD_VALUE_OUTPUT_INDEX));
    const gert::Shape *queryShape = context->GetInputShape(static_cast<size_t>(InputIndexBackward::QUERY_INDEX));
    const gert::Shape *keyShape = context->GetInputShape(static_cast<size_t>(InputIndexBackward::KEY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, gradQueryOutputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradKeyOutputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradValueOutputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    gert::Shape *gradQueryShape = context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_QUERY_INDEX));
    gert::Shape *gradKeyShape = context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_KEY_INDEX));
    gert::Shape *gradValueShape = context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_VALUE_INDEX));
    gert::Shape *gradEncoderQueryShape =
        context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_QUERY_INDEX));
    gert::Shape *gradEncoderKeyShape =
        context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_KEY_INDEX));
    gert::Shape *gradEncoderValueShape =
        context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_VALUE_INDEX));

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto normType = attrs->GetInt(static_cast<size_t>(AttrIndexBackward::NORM_TYPE_INDEX));
    auto normAddedType = attrs->GetInt(static_cast<size_t>(AttrIndexBackward::NORM_ADDED_TYPE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, normType);
    OP_CHECK_NULL_WITH_CONTEXT(context, normAddedType);

    // check required Input tensor And Set Shape
    if (queryShape->GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(context, "Query and Key must be 4D tensors(B, S, H, D).");
        return ge::GRAPH_FAILED;
    }
    int64_t batch = queryShape->GetDim(BATCH_DIM);
    int64_t head = queryShape->GetDim(HEAD_DIM);
    int64_t dim = queryShape->GetDim(DIM_DIM);
    int64_t querySeq = queryShape->GetDim(SEQ_DIM);
    int64_t keySeq = 0;
    int64_t encoderQuerySeq = 0;
    int64_t encoderKeySeq = 0;
    int64_t concatQuerySeq = 0;
    int64_t concatKeySeq = 0;
    int64_t concatValueSeq = 0;

    if (CheckTransposeShape(context, gradQueryOutputShape, batch, head, dim, concatQuerySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckTransposeShape(context, gradKeyOutputShape, batch, head, dim, concatKeySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckTransposeShape(context, gradValueOutputShape, batch, head, dim, concatValueSeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckShape(context, queryShape, batch, head, dim, querySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckShape(context, keyShape, batch, head, dim, keySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *encoderQueryShape =
        context->GetInputShape(static_cast<size_t>(InputIndexBackward::ENCODER_QUERY_INDEX));
    const gert::Shape *encoderKeyShape =
        context->GetInputShape(static_cast<size_t>(InputIndexBackward::ENCODER_KEY_INDEX));

    if (encoderQueryShape != nullptr &&
        CheckShape(context, encoderQueryShape, batch, head, dim, encoderQuerySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (encoderKeyShape != nullptr &&
        CheckShape(context, encoderKeyShape, batch, head, dim, encoderKeySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (querySeq + encoderQuerySeq != concatQuerySeq || keySeq + encoderKeySeq != concatKeySeq ||
        keySeq + encoderKeySeq != concatValueSeq) {
        OP_LOGE(context,
                "Seq sum of (Q, EncoderQ), (K, Encoderk), (V, EncoderV) should equal to it of gradQueryOutput, "
                "gradKeyOutput, gradValueOutput");
    }

    *gradQueryShape = {batch, querySeq, head, dim}; // B, S, H, D
    *gradKeyShape = {batch, keySeq, head, dim};     // B, S, H, D
    *gradValueShape = {batch, keySeq, head, dim};   // B, S, H, D

    if (encoderQueryShape != nullptr) {
        *gradEncoderQueryShape = {batch, encoderQuerySeq, head, dim}; // B, S, H, D
    }

    if (encoderKeyShape != nullptr) {
        *gradEncoderKeyShape = {batch, encoderKeySeq, head, dim};   // B, S, H, D
        *gradEncoderValueShape = {batch, encoderKeySeq, head, dim}; // B, S, H, D
    }

    if (*normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
        gert::Shape *normQueryWeightShape =
            context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_QUERY_WEIGHT_INDEX));
        gert::Shape *normQueryBiasShape =
            context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_QUERY_BIAS_INDEX));
        gert::Shape *normKeyWeightShape =
            context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_KEY_WEIGHT_INDEX));
        gert::Shape *normKeyBiasShape =
            context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_KEY_BIAS_INDEX));
        *normQueryWeightShape = {dim};
        *normQueryBiasShape = {dim};
        *normKeyWeightShape = {dim};
        *normKeyBiasShape = {dim};
    }

    if (*normAddedType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
        gert::Shape *normAddedQueryWeightShape =
            context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_QUERY_WEIGHT_INDEX));
        gert::Shape *normAddedQueryBiasShape =
            context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_QUERY_BIAS_INDEX));
        gert::Shape *normAddedKeyWeightShape =
            context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_KEY_WEIGHT_INDEX));
        gert::Shape *normAddedKeyBiasShape =
            context->GetOutputShape(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_KEY_BIAS_INDEX));
        *normAddedQueryWeightShape = {dim};
        *normAddedQueryBiasShape = {dim};
        *normAddedKeyWeightShape = {dim};
        *normAddedKeyBiasShape = {dim};
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4NormRopeConcatGrad(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("InferDataType4NormRopeConcatGrad", "context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Enter InferDataType4NormRopeConcatGrad.");

    auto inputDtype = context->GetInputDataType(static_cast<size_t>(InputIndexBackward::QUERY_INDEX));
    context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_QUERY_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_KEY_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_VALUE_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_QUERY_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_KEY_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_VALUE_INDEX), inputDtype);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto normType = attrs->GetInt(static_cast<size_t>(AttrIndexBackward::NORM_TYPE_INDEX));
    auto normAddedType = attrs->GetInt(static_cast<size_t>(AttrIndexBackward::NORM_ADDED_TYPE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, normType);
    OP_CHECK_NULL_WITH_CONTEXT(context, normAddedType);
    if (*normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
        context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_QUERY_WEIGHT_INDEX), inputDtype);
        context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_QUERY_BIAS_INDEX), inputDtype);
        context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_KEY_WEIGHT_INDEX), inputDtype);
        context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_KEY_BIAS_INDEX), inputDtype);
    }
    if (*normAddedType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
        context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_QUERY_WEIGHT_INDEX),
                                   inputDtype);
        context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_QUERY_BIAS_INDEX),
                                   inputDtype);
        context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_KEY_WEIGHT_INDEX),
                                   inputDtype);
        context->SetOutputDataType(static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_KEY_BIAS_INDEX),
                                   inputDtype);
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NormRopeConcatGrad).InferShape(InferShape4NormRopeConcatGrad).InferDataType(InferDataType4NormRopeConcatGrad);
} // namespace ops
