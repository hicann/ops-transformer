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
 * \file sparse_flash_mla_softmax_l1_norm_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t SPARSE_INDICES_INDEX = 3;
constexpr uint32_t SOFTMAX_L1_NORM_INDEX = 0;
constexpr uint32_t ATTR_MAX_SEQLEN_K_INDEX = 1;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 4;

static ge::graphStatus InferShapeSparseFlashMlaSoftmaxL1Norm(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashMlaSoftmaxL1Norm", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter SparseFlashMlaSoftmaxL1Norm InferShape impl.");
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *layoutQPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutQPtr);
    std::string layoutQStr(layoutQPtr);
    OP_CHECK_IF(layoutQStr != "BSND" && layoutQStr != "TND",
                OP_LOGE(context, "layout_q only supports BSND or TND, but got %s.", layoutQStr.c_str()),
                return ge::GRAPH_FAILED);
    const int64_t *maxSeqlenKPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_K_INDEX);
    int64_t maxSeqlenK = (maxSeqlenKPtr != nullptr) ? *maxSeqlenKPtr : 0;

    gert::Shape *outShape = context->GetOutputShape(SOFTMAX_L1_NORM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    const gert::Shape *sparseIndiceShape = context->GetOptionalInputShape(static_cast<size_t>(SPARSE_INDICES_INDEX));
    if (layoutQStr == "BSND") {
        OP_CHECK_IF(queryShape->GetDimNum() != 4,
                    OP_LOGE(context, "BSND layout, query dim num (%zu) must be 4!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(keyShape->GetDimNum() != 4,
                    OP_LOGE(context, "BSND layout, key dim num (%zu) must be 4!", keyShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        outShape->SetDimNum(4);
        outShape->SetDim(0, queryShape->GetDim(0)); // B
        outShape->SetDim(1, queryShape->GetDim(1)); // S1
        outShape->SetDim(2, keyShape->GetDim(2));   // N2

        if (sparseIndiceShape != nullptr && sparseIndiceShape->GetShapeSize() != 0) {
            outShape->SetDim(3, sparseIndiceShape->GetDim(3)); // K
        } else {
            outShape->SetDim(3, keyShape->GetDim(1)); // S2
        }
    } else { // TND
        OP_CHECK_IF(queryShape->GetDimNum() != 3,
                    OP_LOGE(context, "TND layout, query dim num (%zu) must be 3!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(keyShape->GetDimNum() != 3,
                    OP_LOGE(context, "TND layout, key dim num (%zu) must be 3!", keyShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        outShape->SetDimNum(3);
        outShape->SetDim(0, queryShape->GetDim(0)); // T1
        outShape->SetDim(1, keyShape->GetDim(0));   // T2
        if (sparseIndiceShape != nullptr && sparseIndiceShape->GetShapeSize() != 0) {
            outShape->SetDim(2, sparseIndiceShape->GetDim(2)); // K
        } else {
            outShape->SetDim(2, maxSeqlenK); // S2
        }
    }

    OP_LOGD(context->GetNodeName(), "SparseFlashMlaSoftmaxL1Norm InferShape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSparseFlashMlaSoftmaxL1Norm(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashMlaSoftmaxL1Norm", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter SparseFlashMlaSoftmaxL1Norm InferDataType impl.");
    context->SetOutputDataType(SOFTMAX_L1_NORM_INDEX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "SparseFlashMlaSoftmaxL1Norm InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashMlaSoftmaxL1Norm)
    .InferShape(InferShapeSparseFlashMlaSoftmaxL1Norm)
    .InferDataType(InferDataTypeSparseFlashMlaSoftmaxL1Norm);
} // namespace ops
