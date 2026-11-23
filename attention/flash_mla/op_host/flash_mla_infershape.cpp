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
 * \file flash_mla_infershape.cpp
 * \brief FlashMla算子InferShape实现
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

namespace ops {

static constexpr size_t ATTR_IDX_HEAD_DIM_V = 0U;
static constexpr size_t ATTR_IDX_LAYOUT_Q = 7U;
static constexpr size_t ATTR_IDX_LAYOUT_KV = 8U;
static constexpr size_t ATTR_IDX_LAYOUT_OUT = 9U;
static constexpr size_t ATTR_IDX_RETURN_SOFTMAX_LSE = 10U;

static constexpr size_t INPUT_IDX_Q = 0U;
static constexpr size_t OUTPUT_IDX_ATTN_OUT = 0U;
static constexpr size_t OUTPUT_IDX_SOFTMAX_LSE = 1U;

ge::graphStatus InferShapeFlashMla(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context, "FlashMla InferShape start.");

    const gert::Shape* qShape = context->GetInputShape(INPUT_IDX_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* layoutQ = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_Q);
    const char* layoutKv = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_KV);
    const char* layoutOut = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_OUT);
    OP_CHECK_IF(layoutQ == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "layout_q",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(layoutKv == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "layout_kv",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(layoutOut == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "layout_out",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);

    auto headDimVPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_HEAD_DIM_V);
    OP_CHECK_IF(headDimVPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "head_dim_v",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    int64_t headDimV = *headDimVPtr;

    auto returnSoftmaxLsePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_RETURN_SOFTMAX_LSE);
    OP_CHECK_IF(returnSoftmaxLsePtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "return_softmax_lse",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    int64_t returnSoftmaxLse = *returnSoftmaxLsePtr;

    if (headDimV <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "head_dim_v", std::to_string(headDimV).c_str(),
                                              "The value of head_dim_v must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    const std::string layoutQStr(layoutQ);
    const std::string layoutKvStr(layoutKv);
    const std::string layoutOutStr(layoutOut);

    OP_LOGI(context, "FlashMla InferShape: layoutQ=%s, layoutKv=%s, layoutOut=%s, headDimV=%ld, returnLSE=%ld.",
            layoutQStr.c_str(), layoutKvStr.c_str(), layoutOutStr.c_str(), headDimV, returnSoftmaxLse);

    int64_t batchSize = 1;
    int64_t numHeadsQ = 0;
    int64_t seqLenQ = 0;
    bool isTND = false;

    if (layoutQStr == "BSND") {
        if (qShape->GetDimNum() != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "q",
                                                     std::to_string(qShape->GetDimNum()).c_str(),
                                                     "The shape dim of q must be 4 when layout_q is BSND");
            return ge::GRAPH_FAILED;
        }
        batchSize = qShape->GetDim(0);
        seqLenQ = qShape->GetDim(1);
        numHeadsQ = qShape->GetDim(2);
    } else if (layoutQStr == "BNSD") {
        if (qShape->GetDimNum() != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "q",
                                                     std::to_string(qShape->GetDimNum()).c_str(),
                                                     "The shape dim of q must be 4 when layout_q is BNSD");
            return ge::GRAPH_FAILED;
        }
        batchSize = qShape->GetDim(0);
        numHeadsQ = qShape->GetDim(1);
        seqLenQ = qShape->GetDim(2);
    } else if (layoutQStr == "TND") {
        if (qShape->GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "q",
                                                     std::to_string(qShape->GetDimNum()).c_str(),
                                                     "The shape dim of q must be 3 when layout_q is TND");
            return ge::GRAPH_FAILED;
        }
        seqLenQ = qShape->GetDim(0); // T = total tokens
        numHeadsQ = qShape->GetDim(1);
        isTND = true;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_q", layoutQStr.c_str(),
                                              "The value of layout_q must be in BSND/BNSD/TND");
        return ge::GRAPH_FAILED;
    }

    if (layoutOutStr != layoutQStr && !(layoutQStr == "TND" && layoutOutStr == "NTD")) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_out", layoutOutStr.c_str(),
                                              "layout_out must equal layout_q, or be NTD when layout_q is TND");
        return ge::GRAPH_FAILED;
    }

    if (layoutKvStr != "PA_BBND" && layoutKvStr != "PA_BNBD" && layoutKvStr != "PA_NZ") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_kv", layoutKvStr.c_str(),
                                              "The value of layout_kv must be in PA_BBND/PA_BNBD/PA_NZ");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* attnOutShape = context->GetOutputShape(OUTPUT_IDX_ATTN_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, attnOutShape);

    if (layoutOutStr == "BSND") {
        attnOutShape->SetDimNum(4);
        attnOutShape->SetDim(0, batchSize);
        attnOutShape->SetDim(1, seqLenQ);
        attnOutShape->SetDim(2, numHeadsQ);
        attnOutShape->SetDim(3, headDimV);
    } else if (layoutOutStr == "BNSD") {
        attnOutShape->SetDimNum(4);
        attnOutShape->SetDim(0, batchSize);
        attnOutShape->SetDim(1, numHeadsQ);
        attnOutShape->SetDim(2, seqLenQ);
        attnOutShape->SetDim(3, headDimV);
    } else if (layoutOutStr == "TND") {
        attnOutShape->SetDimNum(3);
        attnOutShape->SetDim(0, seqLenQ); // T总token数
        attnOutShape->SetDim(1, numHeadsQ);
        attnOutShape->SetDim(2, headDimV);
    } else if (layoutOutStr == "NTD") {
        attnOutShape->SetDimNum(3);
        attnOutShape->SetDim(0, numHeadsQ);
        attnOutShape->SetDim(1, seqLenQ);
        attnOutShape->SetDim(2, headDimV);
    }

    gert::Shape* lseShape = context->GetOutputShape(OUTPUT_IDX_SOFTMAX_LSE);
    if (lseShape != nullptr) {
        if (returnSoftmaxLse != 0) {
            if (isTND) {
                lseShape->SetDimNum(2);
                lseShape->SetDim(0, numHeadsQ);
                lseShape->SetDim(1, seqLenQ);
            } else {
                lseShape->SetDimNum(3);
                lseShape->SetDim(0, batchSize);
                lseShape->SetDim(1, numHeadsQ);
                lseShape->SetDim(2, seqLenQ);
            }
        } else {
            lseShape->SetDimNum(1);
            lseShape->SetDim(0, 0);
        }
    }

    OP_LOGI(context, "FlashMla InferShape done. attnOut dims=%zu.", attnOutShape->GetDimNum());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashMla(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto qDtype = context->GetInputDataType(INPUT_IDX_Q);
    context->SetOutputDataType(OUTPUT_IDX_ATTN_OUT, qDtype);
    context->SetOutputDataType(OUTPUT_IDX_SOFTMAX_LSE, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashMla).InferShape(InferShapeFlashMla).InferDataType(InferDataTypeFlashMla);

} // namespace ops
