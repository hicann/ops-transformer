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
 * \file norm_rope_concat_proto.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "norm_rope_concat_base_host.h"

using namespace ge;
using namespace nrc;
namespace ops {
static ge::graphStatus CheckShape(gert::InferShapeContext *context, const gert::Shape *shape, int64_t batch, int64_t head,
                                  int64_t dim, int64_t &seq)
{
    if (shape->GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "Input must be 4D tensors(B, S, H, D).");
        return ge::GRAPH_FAILED;
    }
    if (shape->GetDim(BATCH_DIM) != batch || shape->GetDim(HEAD_DIM) != head || shape->GetDim(DIM_DIM) != dim) {
        OP_LOGE(context->GetNodeName(), "Input must have the same batch, head and dim as Query.");
        return ge::GRAPH_FAILED;
    }
    seq = shape->GetDim(SEQ_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4NormRopeConcat(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("NormRopeConcat", "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferShape4NormRopeConcat.");

    const gert::Shape *queryShape = context->GetInputShape(static_cast<size_t>(InputIndexForward::QUERY_INDEX));
    const gert::Shape *keyShape = context->GetInputShape(static_cast<size_t>(InputIndexForward::KEY_INDEX));
    const gert::Shape *valueShape = context->GetInputShape(static_cast<size_t>(InputIndexForward::VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    gert::Shape *queryOutputShape =
        context->GetOutputShape(static_cast<size_t>(OutputIndexForward::QUERY_OUTPUT_INDEX));
    gert::Shape *keyOutputShape = context->GetOutputShape(static_cast<size_t>(OutputIndexForward::KEY_OUTPUT_INDEX));
    gert::Shape *valueOutputShape =
        context->GetOutputShape(static_cast<size_t>(OutputIndexForward::VALUE_OUTPUT_INDEX));

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto normType = attrs->GetInt(static_cast<size_t>(AttrIndexForward::NORM_TYPE_INDEX));
    auto normAddedType = attrs->GetInt(static_cast<size_t>(AttrIndexForward::NORM_ADDED_TYPE_INDEX));
    auto isTraining = attrs->GetBool(static_cast<size_t>(AttrIndexForward::IS_TRAINING_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, normType);
    OP_CHECK_NULL_WITH_CONTEXT(context, normAddedType);
    OP_CHECK_NULL_WITH_CONTEXT(context, isTraining);

    // check required Input tensor And Set Shape
    if (queryShape->GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "Query and Key must be 4D tensors(B, S, H, D).");
        return ge::GRAPH_FAILED;
    }
    int64_t batch = queryShape->GetDim(BATCH_DIM);
    int64_t head = queryShape->GetDim(HEAD_DIM);
    int64_t dim = queryShape->GetDim(DIM_DIM);
    int64_t querySeq = queryShape->GetDim(SEQ_DIM);
    int64_t keySeq = 0;
    int64_t valueSeq = 0;
    int64_t encoderQuerySeq = 0;
    int64_t encoderKeySeq = 0;
    int64_t encoderValueSeq = 0;

    if (CheckShape(context, keyShape, batch, head, dim, keySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShape(context, valueShape, batch, head, dim, valueSeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *encoderQueryShape =
        context->GetInputShape(static_cast<size_t>(InputIndexForward::ENCODER_QUERY_INDEX));
    const gert::Shape *encoderKeyShape =
        context->GetInputShape(static_cast<size_t>(InputIndexForward::ENCODER_KEY_INDEX));
    const gert::Shape *encoderValueShape =
        context->GetInputShape(static_cast<size_t>(InputIndexForward::ENCODER_VALUE_INDEX));
    if (encoderQueryShape != nullptr &&
        CheckShape(context, encoderQueryShape, batch, head, dim, encoderQuerySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (encoderKeyShape != nullptr &&
        CheckShape(context, encoderKeyShape, batch, head, dim, encoderKeySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (encoderValueShape != nullptr &&
        CheckShape(context, encoderValueShape, batch, head, dim, encoderValueSeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    *queryOutputShape = {batch, head, querySeq + encoderQuerySeq, dim}; // B, H, S, D
    *keyOutputShape = {batch, head, keySeq + encoderKeySeq, dim};       // B, H, S, D
    *valueOutputShape = {batch, head, valueSeq + encoderValueSeq, dim}; // B, H, S, D

    if (*isTraining) {
        if (*normType == static_cast<int64_t>(NormType::LAYER_NORM) ||
            *normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
            gert::Shape *normQueryMeanShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_QUERY_MEAN_INDEX));
            gert::Shape *normQueryRstdShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_QUERY_RSTD_INDEX));
            gert::Shape *normKeyMeanShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_KEY_MEAN_INDEX));
            gert::Shape *normKeyRstdShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_KEY_RSTD_INDEX));
            *normQueryMeanShape = {batch, querySeq, head, 1};
            *normQueryRstdShape = {batch, querySeq, head, 1};
            *normKeyMeanShape = {batch, keySeq, head, 1};
            *normKeyRstdShape = {batch, keySeq, head, 1};
        }
        if (*normAddedType == static_cast<int64_t>(NormType::LAYER_NORM) ||
            *normAddedType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
            gert::Shape *normAddedQueryMeanShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_MEAN_INDEX));
            gert::Shape *normAddedQueryRstdShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_RSTD_INDEX));
            gert::Shape *normAddedKeyMeanShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_MEAN_INDEX));
            gert::Shape *normAddedKeyRstdShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_RSTD_INDEX));
            *normAddedQueryMeanShape = {batch, encoderQuerySeq, head, 1};
            *normAddedQueryRstdShape = {batch, encoderQuerySeq, head, 1};
            *normAddedKeyMeanShape = {batch, encoderKeySeq, head, 1};
            *normAddedKeyRstdShape = {batch, encoderKeySeq, head, 1};
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4NormRopeConcat(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("NormRopeConcat", "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferDataType4NormRopeConcat.");

    auto inputDtype = context->GetInputDataType(static_cast<size_t>(InputIndexForward::QUERY_INDEX));
    context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::QUERY_OUTPUT_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::KEY_OUTPUT_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::VALUE_OUTPUT_INDEX), inputDtype);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto normType = attrs->GetInt(static_cast<size_t>(AttrIndexForward::NORM_TYPE_INDEX));
    auto normAddedType = attrs->GetInt(static_cast<size_t>(AttrIndexForward::NORM_ADDED_TYPE_INDEX));
    auto isTraining = attrs->GetBool(static_cast<size_t>(AttrIndexForward::IS_TRAINING_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, normType);
    OP_CHECK_NULL_WITH_CONTEXT(context, normAddedType);
    OP_CHECK_NULL_WITH_CONTEXT(context, isTraining);
    if (*isTraining) {
        if (*normType == static_cast<int64_t>(NormType::LAYER_NORM) ||
            *normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_QUERY_MEAN_INDEX), ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_QUERY_RSTD_INDEX), ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_KEY_MEAN_INDEX), ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_KEY_RSTD_INDEX), ge::DT_FLOAT);
        }
        if (*normAddedType == static_cast<int64_t>(NormType::LAYER_NORM) ||
            *normAddedType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_MEAN_INDEX),
                                       ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_RSTD_INDEX),
                                       ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_MEAN_INDEX),
                                       ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_RSTD_INDEX),
                                       ge::DT_FLOAT);
        }
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NormRopeConcat).InferShape(InferShape4NormRopeConcat).InferDataType(InferDataType4NormRopeConcat);
} // namespace ops
