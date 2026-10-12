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
 * \file nsa_selected_attention_grad_proto.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
namespace nsa {

enum class InputIndex : uint32_t {
    QUERY = 0,
    KEY,
    VALUE,
    ATTENTION_OUT,
    ATTENTION_OUT_GRAD,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    TOPK_INDICES,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    ATTEN_MASK
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DK,
    DV
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    SELECTED_BLOCK_COUNT,
    SELECTED_BLOCK_SIZE,
    HEAD_NUM, // N1
    INPUT_LAYOUT
};

ge::graphStatus InferShape4NsaSelectedAttentionGrad(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, "context");
    OP_LOGD(context, "Enter InferShape4NsaSelectedAttentionGrad.");

    const gert::Shape* queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    const gert::Shape* keyShape = context->GetInputShape(static_cast<size_t>(InputIndex::KEY));
    const gert::Shape* valueShape = context->GetInputShape(static_cast<size_t>(InputIndex::VALUE));
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    auto attrs = context->GetAttrs();
    auto scaleValue = attrs->GetInt(static_cast<size_t>(AttrIndex::SCALE_VALUE));
    auto selectedBlockCount = attrs->GetInt(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_COUNT));
    auto selectedBlockSize = attrs->GetInt(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    auto headNum = attrs->GetInt(static_cast<size_t>(AttrIndex::HEAD_NUM));
    const char* inputLayout = attrs->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValue);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedBlockCount);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedBlockSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNum);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);

    std::string inputLayoutStr = std::string(inputLayout);
    for (auto& c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "TND") {
        OP_LOGE(context, "The inputLayout should be TND(case-insensitive), but got %s.", inputLayoutStr.c_str());
        return GRAPH_FAILED;
    }

    gert::Shape* dqShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ));
    gert::Shape* dkShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DK));
    gert::Shape* dvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DV));
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dkShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dvShape);
    *dqShape = *queryShape;
    *dkShape = *keyShape;
    *dvShape = *valueShape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4NsaSelectedAttentionGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, "context");
    OP_LOGD(context, "Enter InferDataType4NsaSelectedAttentionGrad.");

    auto dtype = context->GetInputDataType(static_cast<size_t>(InputIndex::QUERY));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DK), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DV), dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaSelectedAttentionGrad)
    .InferShape(InferShape4NsaSelectedAttentionGrad)
    .InferDataType(InferDataType4NsaSelectedAttentionGrad);
} // namespace nsa
} // namespace ops
