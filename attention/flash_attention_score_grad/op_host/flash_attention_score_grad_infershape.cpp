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
 * \file flash_attention_score_grad_proto.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

static const uint64_t DIM_NUM_2 = 2;
static const uint64_t INDEX_OUT_2 = 2;
static const uint64_t INDEX_OUT_3 = 3;
static const uint64_t INDEX_OUT_4 = 4;
static const uint64_t INDEX_OUT_5 = 5;
static const uint64_t INDEX_OUT_6 = 6;
static const uint64_t INDEX_OUTDTYPE = 11;
ge::graphStatus InferShape4FlashAttentionScoreGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"FlashAttentionScoreGrad", "InferShape: context is null"})),
        return ge::GRAPH_FAILED);
    OP_LOGI(context, "Enter FlashAttentionScoreGrad runtime infershape impl.");
    const gert::Shape* queryShape = context->GetInputShape(0);
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("FlashAttentionScoreGrad", "query",
                                                         "required shape or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(1);
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("FlashAttentionScoreGrad", "key",
                                                         "required shape or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* valueShape = context->GetInputShape(2);
    OP_CHECK_IF(valueShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("FlashAttentionScoreGrad", "value",
                                                         "required shape or attribute is missing"),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("FlashAttentionScoreGrad", "attributes",
                                                         "required attributes are missing"),
                return ge::GRAPH_FAILED);

    auto headNum = attrs->GetInt(4); // N1
    OP_CHECK_IF(headNum == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("FlashAttentionScoreGrad", "head_num",
                                                         "required shape or attribute is missing"),
                return ge::GRAPH_FAILED);
    const char* inputLayout = attrs->GetAttrPointer<char>(5);
    OP_CHECK_IF(inputLayout == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("FlashAttentionScoreGrad", "input_layout",
                                                         "required shape or attribute is missing"),
                return ge::GRAPH_FAILED);
    std::string inputLayoutStr = std::string(inputLayout) == "" ? "empty" : std::string(inputLayout);
    for (auto& c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "BSH" && inputLayoutStr != "SBH" && inputLayoutStr != "BSND" && inputLayoutStr != "BNSD" &&
        inputLayoutStr != "TND") {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayout,
                                   "BSH or SBH or BSND or BNSD or TND");
        return GRAPH_FAILED;
    }

    int64_t b = 0;
    int64_t sQ = 0;
    int64_t sKv = 0;
    int64_t t = 0;
    if (inputLayoutStr == "BSH" || inputLayoutStr == "BSND") {
        b = queryShape->GetDim(0);
        sQ = queryShape->GetDim(1);
        sKv = keyShape->GetDim(1);
    } else if (inputLayoutStr == "SBH") {
        b = queryShape->GetDim(1);
        sQ = queryShape->GetDim(0);
        sKv = keyShape->GetDim(0);
    } else if (inputLayoutStr == "TND") {
        t = queryShape->GetDim(0);
    } else {
        // BNSD
        b = queryShape->GetDim(0);
        sQ = queryShape->GetDim(DIM_NUM_2);
        sKv = keyShape->GetDim(DIM_NUM_2);
    }
    OP_LOGI(context, "B=%ld, N=%ld, T=%ld, Sq=%ld, Skv=%ld, inputLayout=%s", b, *headNum, t, sQ, sKv,
            inputLayoutStr.c_str());

    gert::Shape* dqShape = context->GetOutputShape(0);
    OP_CHECK_IF(
        dqShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"FlashAttentionScoreGrad", "InferShape: dqShape is null"})),
        return ge::GRAPH_FAILED);
    *dqShape = *queryShape;

    gert::Shape* dkShape = context->GetOutputShape(1);
    OP_CHECK_IF(
        dkShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"FlashAttentionScoreGrad", "InferShape: dkShape is null"})),
        return ge::GRAPH_FAILED);
    *dkShape = *keyShape;

    gert::Shape* dvShape = context->GetOutputShape(2);
    OP_CHECK_IF(
        dvShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"FlashAttentionScoreGrad", "InferShape: dvShape is null"})),
        return ge::GRAPH_FAILED);
    *dvShape = *valueShape;

    // dpse output
    gert::Shape* dpseShape = context->GetOutputShape(3);
    OP_CHECK_IF(dpseShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"FlashAttentionScoreGrad", "InferShape: dpseShape is null"})),
                return ge::GRAPH_FAILED);

    gert::Shape* dqRopeShape = context->GetOutputShape(4);

    gert::Shape* dkRopeShape = context->GetOutputShape(5);

    gert::Shape* dsinkShape = context->GetOutputShape(6);

    const gert::Shape* pseShape = context->GetOptionalInputShape(4);
    if (pseShape != nullptr && pseShape->GetShapeSize() != 0) {
        OP_LOGD(context, "pse_shift is not nullptr");
        *dpseShape = *pseShape;
    } else {
        OP_LOGD(context, "pse_shift is nullptr");
        dpseShape->SetDimNum(1);
        dpseShape->SetDim(0, 0);
    }
    const gert::Shape* queryRopeShape = context->GetOptionalInputShape(22);
    if (queryRopeShape != nullptr) {
        OP_LOGD(context, "queryRope is not nullptr");
        *dqRopeShape = *queryRopeShape;
    }
    const gert::Shape* keyRopeShape = context->GetOptionalInputShape(23);
    if (keyRopeShape != nullptr) {
        OP_LOGD(context, "keyRope is not nullptr");
        *dkRopeShape = *keyRopeShape;
    }
    const gert::Shape* sinkShape = context->GetOptionalInputShape(24);
    if (sinkShape != nullptr && sinkShape->GetShapeSize() != 0) {
        OP_LOGD(context, "sink is not nullptr");
        *dsinkShape = *sinkShape;
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4FlashAttentionScoreGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"FlashAttentionScoreGrad", "InferShape: context is null"})),
        return ge::GRAPH_FAILED);
    OP_LOGI(context, "Enter FlashAttentionScoreGrad infer data type impl.");

    auto dtype = context->GetInputDataType(0);
    if (dtype == DT_FLOAT8_E5M2 || dtype == DT_FLOAT8_E4M3FN || dtype == DT_HIFLOAT8) {
        auto attrs = context->GetAttrs();
        OP_CHECK_IF(attrs == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("FlashAttentionScoreGrad", "attributes",
                                                             "required attributes are missing"),
                    return ge::GRAPH_FAILED);

        auto outDtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_OUTDTYPE);
        int64_t outDtype = *outDtypePtr;
        if (outDtype == 0) {
            // dq, outidx:0
            context->SetOutputDataType(0, ge::DT_FLOAT16);
            // dk, outidx:1
            context->SetOutputDataType(1, ge::DT_FLOAT16);
            // dv, outidx:2
            context->SetOutputDataType(INDEX_OUT_2, ge::DT_FLOAT16);
            // dpse, outidx:3
            // 后续针对pse内部生成的场景，Dtype就不能跟随qkv了
            context->SetOutputDataType(INDEX_OUT_3, ge::DT_FLOAT16);
            // dq_rope, outidx:4
            context->SetOutputDataType(INDEX_OUT_4, ge::DT_FLOAT16);
            // dk_rope, outidx:5
            context->SetOutputDataType(INDEX_OUT_5, ge::DT_FLOAT16);
            // dsink, outidx:6
            context->SetOutputDataType(INDEX_OUT_6, ge::DT_FLOAT);
        } else if (outDtype == 1) {
            // dq, outidx:0
            context->SetOutputDataType(0, ge::DT_BF16);
            // dk, outidx:1
            context->SetOutputDataType(1, ge::DT_BF16);
            // dv, outidx:2
            context->SetOutputDataType(INDEX_OUT_2, ge::DT_BF16);
            // dpse, outidx:3
            // 后续针对pse内部生成的场景，Dtype就不能跟随qkv了
            context->SetOutputDataType(INDEX_OUT_3, ge::DT_BF16);
            // dq_rope, outidx:4
            context->SetOutputDataType(INDEX_OUT_4, ge::DT_BF16);
            // dk_rope, outidx:5
            context->SetOutputDataType(INDEX_OUT_5, ge::DT_BF16);
            // dsink, outidx:6
            context->SetOutputDataType(INDEX_OUT_6, ge::DT_FLOAT);
        } else {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "outDtype",
                                                  std::to_string(outDtype).c_str(),
                                                  "The value of outDtype must be 0 or 1");
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    // dq, outidx:0
    context->SetOutputDataType(0, dtype);
    // dk, outidx:1
    context->SetOutputDataType(1, dtype);
    // dv, outidx:2
    context->SetOutputDataType(INDEX_OUT_2, dtype);
    // dpse, outidx:3
    // 后续针对pse内部生成的场景，Dtype就不能跟随qkv了
    context->SetOutputDataType(INDEX_OUT_3, dtype);
    // dq_rope, outidx:4
    context->SetOutputDataType(INDEX_OUT_4, dtype);
    // dk_rope, outidx:5
    context->SetOutputDataType(INDEX_OUT_5, dtype);
    // dsink, outidx:6
    context->SetOutputDataType(INDEX_OUT_6, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttentionScoreGrad)
    .InferShape(InferShape4FlashAttentionScoreGrad)
    .InferDataType(InferDataType4FlashAttentionScoreGrad);

} // namespace ops
