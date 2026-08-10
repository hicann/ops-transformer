/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mixed_quant_sparse_flash_mla_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "mixed_quant_sparse_flash_mla_check.h"

using namespace ge;
using namespace optiling;

namespace ops {
constexpr uint32_t QUERY_INPUT_INDEX = 0;

static int64_t GetKvHeadNum(const gert::Shape *kvShape, const std::string &layoutKv)
{
    if (layoutKv == "TND") {
        return kvShape->GetDim(DIM_IDX_ONE);
    }
    return kvShape->GetDim(DIM_IDX_TWO);
}

static std::vector<int64_t> ToVectorFunc(const gert::Shape *shape)
{
    size_t shapeSize = shape->GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape->GetDim(i);
    }
    return shapeVec;
}

static std::string ToStringFunc(const gert::Shape *shape)
{
    std::ostringstream oss;
    auto v = ToVectorFunc(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

ge::graphStatus InferShapeMixedQuantSparseFlashMla(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MixedQuantSparseFlashMla", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);
    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const bool *returnSoftmaxLsePtr = attr->GetAttrPointer<bool>(ATTR_RETURN_SOFTMAX_LSE_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;

    const gert::Shape *kvShape = context->GetInputShape(ORI_KV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
    const char *layoutQ = attr->GetStr(ATTR_LAYOUT_Q_INDEX);
    const char *layoutKv = attr->GetStr(ATTR_LAYOUT_KV_INDEX);
    std::string layoutQStr = (layoutQ != nullptr) ? std::string(layoutQ) : "BSND";
    std::string layoutKvStr = (layoutKv != nullptr) ? std::string(layoutKv) : "BSND";

    int64_t kvHeadNum = GetKvHeadNum(kvShape, layoutKvStr);
    OP_CHECK_IF(kvHeadNum <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("MixedQuantSparseFlashMla", "ori_kv or cmp_kv",
            ToStringFunc(kvShape).c_str(),
            "The head num of ori_kv or cmp_kv should be greater than 0 but got " + std::to_string(kvHeadNum)),
        return ge::GRAPH_FAILED);

    if (returnSoftmaxLse) {
        if (layoutQStr == "TND") {
            softmaxLseShape->SetDimNum(DIM_NUM_THREE);
            softmaxLseShape->SetDim(DIM_IDX_ZERO, kvHeadNum);
            softmaxLseShape->SetDim(DIM_IDX_ONE, queryShape->GetDim(DIM_IDX_ZERO));
            softmaxLseShape->SetDim(DIM_IDX_TWO, queryShape->GetDim(DIM_IDX_ONE) / kvHeadNum);
        } else {
            softmaxLseShape->SetDimNum(DIM_NUM_FOUR);
            softmaxLseShape->SetDim(DIM_IDX_ZERO, queryShape->GetDim(DIM_IDX_ZERO));
            softmaxLseShape->SetDim(DIM_IDX_ONE, kvHeadNum);
            softmaxLseShape->SetDim(DIM_IDX_TWO, queryShape->GetDim(DIM_IDX_ONE));
            softmaxLseShape->SetDim(DIM_IDX_THREE, queryShape->GetDim(DIM_IDX_TWO) / kvHeadNum);
        }
    } else {
        softmaxLseShape->SetDimNum(DIM_NUM_ONE);
        softmaxLseShape->SetDim(DIM_IDX_ZERO, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMixedQuantSparseFlashMla(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MixedQuantSparseFlashMla", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MixedQuantSparseFlashMla)
    .InferShape(InferShapeMixedQuantSparseFlashMla)
    .InferDataType(InferDataTypeMixedQuantSparseFlashMla);
} // namespace ops
