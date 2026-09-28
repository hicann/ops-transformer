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
 * \file sparse_flash_attention_grad_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
namespace sfag {

enum class InputIndex : uint32_t {
    QUERY = 0,
    KEY,
    VALUE,
    TOPK_INDICES,
    ATTENTION_OUT_GRAD,
    ATTENTION_OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    Q_ROPE,
    K_ROPE
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DK,
    DV,
    DQ_ROPE,
    DK_ROPE
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    SELECTED_BLOCK_SIZE,
    INPUT_LAYOUT
};

ge::graphStatus InferShape4SparseFlashAttentionGrad(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttentionGrad", "context is nullptr."),
                return ge::GRAPH_FAILED);

    const gert::Shape *queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    OP_CHECK_IF(queryShape == nullptr, OP_LOGE("SparseFlashAttentionGrad", "queryShape is nullptr."),
                return ge::GRAPH_FAILED)
    const gert::Shape *keyShape = context->GetInputShape(static_cast<size_t>(InputIndex::KEY));
    OP_CHECK_IF(keyShape == nullptr, OP_LOGE("SparseFlashAttentionGrad", "keyShape is nullptr."),
                return ge::GRAPH_FAILED)
    const gert::Shape *valueShape = context->GetInputShape(static_cast<size_t>(InputIndex::VALUE));
    OP_CHECK_IF(valueShape == nullptr, OP_LOGE("SparseFlashAttentionGrad", "valueShape is nullptr."),
                return ge::GRAPH_FAILED)
    const gert::Shape *queryRopeShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::Q_ROPE));
    const gert::Shape *keyRopeShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::K_ROPE));

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("SparseFlashAttentionGrad", "attrs is nullptr."), return ge::GRAPH_FAILED)
    auto scaleValue = attrs->GetFloat(static_cast<size_t>(AttrIndex::SCALE_VALUE));
    OP_CHECK_IF(scaleValue == nullptr, OP_LOGE("SparseFlashAttentionGrad", "scaleValue is nullptr."),
                return ge::GRAPH_FAILED)
    auto selectedBlockSize = attrs->GetInt(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    OP_CHECK_IF(selectedBlockSize == nullptr, OP_LOGE("SparseFlashAttentionGrad", "selectedBlockSize is nullptr."),
                return ge::GRAPH_FAILED)
    const char *inputLayout = attrs->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    OP_CHECK_IF(inputLayout == nullptr, OP_LOGE("SparseFlashAttentionGrad", "inputLayout is nullptr."),
                return ge::GRAPH_FAILED)

    gert::Shape *dqShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ));
    OP_CHECK_IF(dqShape == nullptr, OP_LOGE("SparseFlashAttentionGrad", "dqShape is nullptr."), return ge::GRAPH_FAILED)
    gert::Shape *dkShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DK));
    OP_CHECK_IF(dkShape == nullptr, OP_LOGE("SparseFlashAttentionGrad", "dkShape is nullptr."), return ge::GRAPH_FAILED)
    gert::Shape *dvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DV));
    OP_CHECK_IF(dvShape == nullptr, OP_LOGE("SparseFlashAttentionGrad", "dvShape is nullptr."), return ge::GRAPH_FAILED)
    *dqShape = *queryShape;
    *dkShape = *keyShape;
    *dvShape = *valueShape;

    if (queryRopeShape != nullptr) {
        gert::Shape *dqRopeShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ_ROPE));
        OP_CHECK_IF(dqRopeShape == nullptr, OP_LOGE("SparseFlashAttentionGrad", "dqRopeShape is nullptr."),
                    return ge::GRAPH_FAILED)
        *dqRopeShape = *queryRopeShape;
    }
    if (keyRopeShape != nullptr) {
        gert::Shape *dkRopeShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DK_ROPE));
        OP_CHECK_IF(dkRopeShape == nullptr, OP_LOGE("SparseFlashAttentionGrad", "dkRopeShape is nullptr."),
                    return ge::GRAPH_FAILED)
        *dkRopeShape = *keyRopeShape;
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4SparseFlashAttentionGrad(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttentionGrad", "context is nullptr."),
                return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDataType(static_cast<size_t>(InputIndex::QUERY));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DK), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DV), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ_ROPE), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DK_ROPE), dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP(SparseFlashAttentionGrad)
    .InferShape(InferShape4SparseFlashAttentionGrad)
    .InferDataType(InferDataType4SparseFlashAttentionGrad);
} // namespace sfag
} // namespace ops
