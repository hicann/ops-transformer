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
 * \file nsa_selected_attention_infer_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr size_t VALUE_INPUT_INDEX = 2;
constexpr size_t INPUT_LAYOUT_ATTR_INDEX = 0;
static const int64_t HEAD_DIM_V_MAX = 128;

namespace {
void ReportInferShapeFailure(const char* opName, const char* reason)
{
    OP_LOGE_WITHOUT_REPORT(opName, "InferShape failed: %s", reason);
    REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                              (std::vector<const char*>{opName, reason}));
}
} // namespace

ge::graphStatus InferShapeNsaSelectAttentionInfer(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        ReportInferShapeFailure("NsaSelectedAttentionInfer", "InferShape context is missing");
        return ge::GRAPH_FAILED;
    }
    const char* opName = context->GetNodeName();
    if (opName == nullptr) {
        opName = "NsaSelectedAttentionInfer";
    }
    OP_LOGI(opName, "Enter NsaSelectedAttentionInfer runtime infershape impl.");
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_IF(queryShape == nullptr, ReportInferShapeFailure(opName, "query shape is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_IF(keyShape == nullptr, ReportInferShapeFailure(opName, "key shape is missing"), return ge::GRAPH_FAILED);
    const gert::Shape* valueShape = context->GetInputShape(VALUE_INPUT_INDEX);
    OP_CHECK_IF(valueShape == nullptr, ReportInferShapeFailure(opName, "value shape is missing"),
                return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    const auto* queryDesc = context->GetInputDesc(QUERY_INPUT_INDEX);
    OP_CHECK_IF(queryDesc == nullptr, ReportInferShapeFailure(opName, "query descriptor is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs == nullptr, ReportInferShapeFailure(opName, "attribute container is missing"),
                return ge::GRAPH_FAILED);

    const char* inputLayout = attrs->GetAttrPointer<char>(INPUT_LAYOUT_ATTR_INDEX);
    OP_CHECK_IF(inputLayout == nullptr, ReportInferShapeFailure(opName, "input_layout attribute is missing"),
                return ge::GRAPH_FAILED);
    std::string inputLayoutStr = std::string(inputLayout);
    for (auto& c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "BSND" && inputLayoutStr != "BSH" && inputLayoutStr != "TND") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "input_layout", inputLayout,
                                              "Supported layouts are BSND, BSH and TND");
        return GRAPH_FAILED;
    }

    gert::Shape* attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_IF(attentionOutShape == nullptr, ReportInferShapeFailure(opName, "attention_out shape is missing"),
                return ge::GRAPH_FAILED);
    *attentionOutShape = *queryShape;
    if (inputLayoutStr == "BSND") {
        auto shapeD2 = valueShape->GetDim(DIM_NUM_3);
        attentionOutShape->SetDim(DIM_NUM_3, shapeD2);
    } else if (inputLayoutStr == "BSH") {
        if (keyShape->GetDim(DIM_NUM_2) == 0) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "key.shape[2]", "0",
                                                      "The BSH hidden size of key must be nonzero");
            return GRAPH_FAILED;
        }
        if (queryShape->GetDim(DIM_NUM_2) % keyShape->GetDim(DIM_NUM_2) != 0) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName, "query, key",
                std::to_string(queryShape->GetDim(DIM_NUM_2)) + ", " + std::to_string(keyShape->GetDim(DIM_NUM_2)),
                "The BSH hidden size of query must be divisible by the hidden size of key");
            return GRAPH_FAILED;
        }
        auto group = queryShape->GetDim(DIM_NUM_2) / keyShape->GetDim(DIM_NUM_2);
        auto shapeD2 = valueShape->GetDim(DIM_NUM_2) * group;
        attentionOutShape->SetDim(DIM_NUM_2, shapeD2);
    } else if (inputLayoutStr == "TND") {
        attentionOutShape->SetDim(DIM_NUM_2, HEAD_DIM_V_MAX);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeNsaSelectAttentionInfer(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        ReportInferShapeFailure("NsaSelectedAttentionInfer", "InferDataType context is missing");
        return ge::GRAPH_FAILED;
    }
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaSelectedAttentionInfer)
    .InferShape(InferShapeNsaSelectAttentionInfer)
    .InferDataType(InferDataTypeNsaSelectAttentionInfer);
} // namespace ops
