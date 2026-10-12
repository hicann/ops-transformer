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
 * \file rope_quant_kvcache_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
using namespace ge;
namespace ops {
static constexpr int KVCACHE_INPUT_INDEX = 5;
static constexpr int Q_OUTPUT_INDEX = 0;
static constexpr int K_OUTPUT_INDEX = 1;
static constexpr int V_OUTPUT_INDEX = 2;
static constexpr int KCACHE_OUTPUT_INDEX = 3;
static constexpr int VCACHE_OUTPUT_INDEX = 4;
static constexpr int TOTAL_DIM = 4;
static constexpr int THIRD_DIM = 2;
static constexpr int FORTH_DIM = 3;

static ge::graphStatus RopeQuantKvcacheInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qkvShape = context->GetInputShape(0);
    const gert::Shape* cacheShape = context->GetInputShape(KVCACHE_INPUT_INDEX);

    gert::Shape* qOutShape = context->GetOutputShape(Q_OUTPUT_INDEX);
    gert::Shape* kOutShape = context->GetOutputShape(K_OUTPUT_INDEX);
    gert::Shape* vOutShape = context->GetOutputShape(V_OUTPUT_INDEX);
    gert::Shape* kCacheShape = context->GetOutputShape(KCACHE_OUTPUT_INDEX);
    gert::Shape* vCacheShape = context->GetOutputShape(VCACHE_OUTPUT_INDEX);

    auto attrs = context->GetAttrs();
    const gert::ContinuousVector* splitSize = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const int64_t* splitSizeArray = reinterpret_cast<const int64_t*>(splitSize->GetData());

    int64_t batch = cacheShape->GetDim(0);
    int64_t qkvSeqlen = qkvShape->GetDim(1);
    int64_t kvHead = cacheShape->GetDim(THIRD_DIM);
    int64_t hiddenSize = cacheShape->GetDim(FORTH_DIM);
    int64_t qHead = splitSizeArray[0] / hiddenSize;

    qOutShape->SetDimNum(TOTAL_DIM);
    qOutShape->SetDim(0, batch);
    qOutShape->SetDim(1, qkvSeqlen);
    qOutShape->SetDim(THIRD_DIM, qHead);
    qOutShape->SetDim(FORTH_DIM, hiddenSize);

    kOutShape->SetDimNum(TOTAL_DIM);
    kOutShape->SetDim(0, batch);
    kOutShape->SetDim(1, qkvSeqlen);
    kOutShape->SetDim(THIRD_DIM, kvHead);
    kOutShape->SetDim(FORTH_DIM, hiddenSize);

    vOutShape->SetDimNum(TOTAL_DIM);
    vOutShape->SetDim(0, batch);
    vOutShape->SetDim(1, qkvSeqlen);
    vOutShape->SetDim(THIRD_DIM, kvHead);
    vOutShape->SetDim(FORTH_DIM, hiddenSize);

    int64_t cacheDim = cacheShape->GetDimNum();
    kCacheShape->SetDimNum(cacheDim);
    vCacheShape->SetDimNum(cacheDim);
    for (int64_t i = 0; i < cacheDim; i++) {
        kCacheShape->SetDim(i, cacheShape->GetDim(i));
        vCacheShape->SetDim(i, cacheShape->GetDim(i));
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus RopeQuantKvcacheInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType q = context->GetInputDataType(0);
    const ge::DataType afterQuant = context->GetInputDataType(KVCACHE_INPUT_INDEX);
    context->SetOutputDataType(Q_OUTPUT_INDEX, q);
    context->SetOutputDataType(K_OUTPUT_INDEX, q);
    context->SetOutputDataType(V_OUTPUT_INDEX, q);
    context->SetOutputDataType(KCACHE_OUTPUT_INDEX, afterQuant);
    context->SetOutputDataType(VCACHE_OUTPUT_INDEX, afterQuant);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(RopeQuantKvcache)
    .InferShape(RopeQuantKvcacheInferShape)
    .InferDataType(RopeQuantKvcacheInferDataType);
} // namespace ops
