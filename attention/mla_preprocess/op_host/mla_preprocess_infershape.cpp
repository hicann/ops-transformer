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
  * \file mla_preprocess_infershape.cpp
  * \brief
  */

#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;
namespace ops {

constexpr size_t ATTR_CACHE_MODE = 9;

constexpr size_t INPUT_INPUT = 0;
constexpr size_t INPUT_COS = 16;
constexpr size_t INPUT_SIN = 17;
constexpr size_t INPUT_WUK = 18;
constexpr size_t INPUT_KV_CACHE = 19;
constexpr size_t INPUT_KV_CACHE_ROPE = 20;

constexpr size_t OUTPUT_Q_OUT = 0;
constexpr size_t OUTPUT_KV_CACHE_OUT = 1;
constexpr size_t OUTPUT_Q_ROPE_OUT = 2;
constexpr size_t OUTPUT_KR_CACHE_OUT = 3;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr size_t DIM_SHAPE_Q = 512;
constexpr size_t DIM_SHAPE_QR = 64;
constexpr size_t DIM_SHAPE_Q_CACHE = 576;
constexpr int64_t ROPE_HEAD_DIM = 64;

static bool IsDisabledRopeShape(const gert::Shape *shape)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == 0;
}

static ge::graphStatus CheckRopeInputShapes(gert::InferShapeContext *context, int64_t tokenNum)
{
    const gert::Shape *cosShape = context->GetRequiredInputShape(INPUT_COS);
    const gert::Shape *sinShape = context->GetRequiredInputShape(INPUT_SIN);
    OP_CHECK_IF(cosShape == nullptr || sinShape == nullptr,
                OP_LOGE(context->GetNodeName(), "cos and sin must be provided."),
                return ge::GRAPH_FAILED);

    const bool cosDisablesRope = IsDisabledRopeShape(cosShape);
    const bool sinDisablesRope = IsDisabledRopeShape(sinShape);
    OP_CHECK_IF(cosDisablesRope != sinDisablesRope,
                OP_LOGE(context->GetNodeName(), "cos and sin must both have shape [0] or neither have shape [0]."),
                return ge::GRAPH_FAILED);
    if (!cosDisablesRope) {
        OP_CHECK_IF(cosShape->GetDimNum() != 2 || sinShape->GetDimNum() != 2 || cosShape->GetDim(0) != tokenNum ||
                        sinShape->GetDim(0) != tokenNum || cosShape->GetDim(1) != ROPE_HEAD_DIM ||
                        sinShape->GetDim(1) != ROPE_HEAD_DIM,
                    OP_LOGE(context->GetNodeName(), "cos and sin must have shape [tokenNum, 64]."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeMlaPreprocess(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMlaPreprocess.");
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *cacheModePtr = attrs->GetAttrPointer<int64_t>(ATTR_CACHE_MODE);
    if (cacheModePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t cacheMode = *cacheModePtr;

    const gert::Shape *kvCacheShape = context->GetRequiredInputShape(INPUT_KV_CACHE);
    const gert::Shape *kvCacheRopeShape = context->GetRequiredInputShape(INPUT_KV_CACHE_ROPE);
    const gert::Shape *inputShape = context->GetRequiredInputShape(INPUT_INPUT);
    const gert::Shape *wukShape = context->GetRequiredInputShape(INPUT_WUK);

    const int64_t tokenNum = inputShape->GetDim(0);
    const int64_t headNum = wukShape->GetDim(0);

    if (CheckRopeInputShapes(context, tokenNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *qOutShape = context->GetOutputShape(OUTPUT_Q_OUT);
    gert::Shape *kvCacheOutShape = context->GetOutputShape(OUTPUT_KV_CACHE_OUT);

    *kvCacheOutShape = *kvCacheShape;

    qOutShape->SetDimNum(DIM_THREE);
    qOutShape->SetDim(0, tokenNum);
    qOutShape->SetDim(1, headNum);

    if (cacheMode != 0) {
        gert::Shape *qRopeOutShape = context->GetOutputShape(OUTPUT_Q_ROPE_OUT);
        gert::Shape *krCacheOutShape = context->GetOutputShape(OUTPUT_KR_CACHE_OUT);
        *krCacheOutShape = *kvCacheRopeShape;

        qOutShape->SetDim(DIM_TWO, DIM_SHAPE_Q);

        qRopeOutShape->SetDimNum(DIM_THREE);
        qRopeOutShape->SetDim(0, tokenNum);
        qRopeOutShape->SetDim(1, headNum);
        qRopeOutShape->SetDim(DIM_TWO, DIM_SHAPE_QR);
    } else {
        qOutShape->SetDim(DIM_TWO, DIM_SHAPE_Q_CACHE);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeMlaPreprocess.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMlaPreprocess(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMlaPreprocess.");
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *cacheModePtr = attrs->GetAttrPointer<int64_t>(ATTR_CACHE_MODE);
    if (cacheModePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t cacheMode = *cacheModePtr;

    auto inputDtype = context->GetRequiredInputDataType(INPUT_INPUT);
    auto kvCacheDtype = context->GetRequiredInputDataType(INPUT_KV_CACHE);
    context->SetOutputDataType(OUTPUT_Q_OUT, kvCacheDtype);
    context->SetOutputDataType(OUTPUT_KV_CACHE_OUT, kvCacheDtype);
    if (cacheMode != 0) {
        context->SetOutputDataType(OUTPUT_Q_ROPE_OUT, inputDtype);
        context->SetOutputDataType(OUTPUT_KR_CACHE_OUT, inputDtype);
    }
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMlaPreprocess.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaPreprocess).InferShape(InferShapeMlaPreprocess).InferDataType(InferDataTypeMlaPreprocess);

}  // namespace ops
