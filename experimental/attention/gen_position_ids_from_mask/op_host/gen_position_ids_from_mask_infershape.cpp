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
 * \file gen_position_ids_from_mask_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t MASK_INPUT_INDEX = 0;
constexpr uint32_t POSITION_IDS_OUTPUT_INDEX = 0;
constexpr size_t DIMS_LIMIT = 2;
constexpr size_t DIM_B = 0;
constexpr size_t DIM_S = 1;

ge::graphStatus InferShapeGenPositionIdsFromMask(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GenPositionIdsFromMask", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *maskShape = context->GetInputShape(MASK_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);
    gert::Shape *posShape = context->GetOutputShape(POSITION_IDS_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, posShape);

    const size_t dimNum = maskShape->GetDimNum();
    OP_CHECK_IF(dimNum != DIMS_LIMIT, OP_LOGE("GenPositionIdsFromMask", "attentionMask must be a 2D tensor."),
                return ge::GRAPH_FAILED);

    posShape->SetDimNum(DIMS_LIMIT);
    posShape->SetDim(DIM_B, maskShape->GetDim(DIM_B));
    posShape->SetDim(DIM_S, maskShape->GetDim(DIM_S));
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeGenPositionIdsFromMask(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GenPositionIdsFromMask", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(POSITION_IDS_OUTPUT_INDEX, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(GenPositionIdsFromMask)
    .InferShape(InferShapeGenPositionIdsFromMask)
    .InferDataType(InferDataTypeGenPositionIdsFromMask);
} // namespace ops
