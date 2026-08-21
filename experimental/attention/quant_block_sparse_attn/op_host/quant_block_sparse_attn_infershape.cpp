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
 * \file quant_block_sparse_attn_infershape.cpp
 * \brief QuantBlockSparseAttn infer shape and data type.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "quant_block_sparse_attn_tiling.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
} // namespace

ge::graphStatus InferShapeQuantBlockSparseAttn(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "InferShape: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(optiling::QBSA_QUERY_INDEX);
    if (queryShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: query shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    gert::Shape *attentionOutShape = context->GetOutputShape(optiling::QBSA_ATTENTION_OUT_INDEX);
    if (attentionOutShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: attentionOut shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(kOpName, "InferShape: attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    const std::string layoutQ = optiling::QBSAGetStringAttr(attrs, optiling::QBSA_LAYOUT_Q_ATTR_INDEX, "TND");
    if ((queryShape->GetDimNum() != 3U) || (layoutQ != "TND" && layoutQ != "NTD")) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "query",
                                                 std::to_string(queryShape->GetDimNum()) + "D with layout " + layoutQ,
                                                 "3D with layout TND or 3D with layout NTD");
        return ge::GRAPH_FAILED;
    }
    if (layoutQ == "NTD") {
        attentionOutShape->SetDimNum(3);
        attentionOutShape->SetDim(0, queryShape->GetDim(1)); // T
        attentionOutShape->SetDim(1, queryShape->GetDim(0)); // N
        attentionOutShape->SetDim(2, queryShape->GetDim(2)); // D
    } else {
        *attentionOutShape = *queryShape;
    }

    gert::Shape *softmaxLseShape = context->GetOutputShape(optiling::QBSA_SOFTMAX_LSE_INDEX);
    if (softmaxLseShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: softmaxLse shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const bool *returnSoftmaxLsePtr = attrs->GetAttrPointer<bool>(optiling::QBSA_RETURN_SOFTMAX_LSE_ATTR_INDEX);
    const bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    if (!returnSoftmaxLse) {
        softmaxLseShape->SetDimNum(1);
        softmaxLseShape->SetDim(0, 0);
        return ge::GRAPH_SUCCESS;
    }
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(optiling::QBSA_QUANT_MODE_ATTR_INDEX);
    const uint32_t quantMode =
        (quantModePtr != nullptr) ? static_cast<uint32_t>(*quantModePtr) : optiling::QBSA_QUANT_MODE_FP8;

    softmaxLseShape->SetDimNum(2);
    if (quantMode == optiling::QBSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        softmaxLseShape->SetDim(0, queryShape->GetDim(0)); // T
        softmaxLseShape->SetDim(1, queryShape->GetDim(1)); // N1
    } else if (layoutQ == "NTD") {
        softmaxLseShape->SetDim(0, queryShape->GetDim(0)); // N1
        softmaxLseShape->SetDim(1, queryShape->GetDim(1)); // T
    } else {
        softmaxLseShape->SetDim(0, queryShape->GetDim(1)); // N1
        softmaxLseShape->SetDim(1, queryShape->GetDim(0)); // T
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeQuantBlockSparseAttn(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "InferDataType: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    ge::DataType attentionOutType = context->GetOutputDataType(optiling::QBSA_ATTENTION_OUT_INDEX);
    if (attentionOutType != ge::DT_BF16) {
        attentionOutType = ge::DT_BF16;
    }
    context->SetOutputDataType(optiling::QBSA_ATTENTION_OUT_INDEX, attentionOutType);
    context->SetOutputDataType(optiling::QBSA_SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBlockSparseAttn)
    .InferShape(InferShapeQuantBlockSparseAttn)
    .InferDataType(InferDataTypeQuantBlockSparseAttn);
} // namespace ops
