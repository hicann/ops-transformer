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
 * \file partial_rotary_mul_quant_infershape.cpp
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PARTIAL_ROTARY_MUL_QUANT_OPS_H_
#define OPS_OP_PROTO_INC_PARTIAL_ROTARY_MUL_QUANT_OPS_H_

#include "util/shape_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
const int32_t INPUT_IDX_X = 0;
const int32_t INPUT_IDX_COS = 1;
const int32_t INPUT_IDX_SIN = 2;
const int32_t INPUT_IDX_SCALE = 3;
const int32_t INDEX_OUTPUT_Y = 0;
const int32_t INDEX_OUTPUT_Y_SCALE = 1;
const int32_t INDEX_ATTR_Y_DTYPE = 0;
const int32_t INDEX_ATTR_QUANT_MODE = 1;
const int32_t INDEX_ATTR_ROUND_MODE = 2;
const int32_t INDEX_ATTR_AXIS = 3;
const int32_t INDEX_ATTR_ROTARY_MODE = 4;
const int32_t INDEX_ATTR_SLICE = 5;
const int64_t ALIGN_NUM_FOR_HIFLOAT4 = 1;
const int64_t BLOCKSIZE4HIFLOAT4 = 64;
const int64_t UNKNOWN_DIM_VALUE_ = -1;

static ge::graphStatus InferShape4PartialRotaryMulQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShape4PartialRotaryMulQuant.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* cosShape = context->GetInputShape(INPUT_IDX_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosShape);

    const gert::Shape* sinShape = context->GetInputShape(INPUT_IDX_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinShape);

    auto yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto scaleShape = context->GetOutputShape(INDEX_OUTPUT_Y_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, scale shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*scaleShape);
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *xShape;

    int64_t xDimSize = xShape->GetDimNum();
    int64_t dimSize = 0;
    if (xShape->GetDim(xDimSize - 1) == UNKNOWN_DIM_VALUE_) { // only support last quant now
        dimSize = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize = Ops::Base::CeilDiv(xShape->GetDim(xDimSize - 1), BLOCKSIZE4HIFLOAT4);
    }

    *scaleShape = *xShape;
    scaleShape->SetDim(xDimSize - 1, dimSize);
    scaleShape->AppendDim(ALIGN_NUM_FOR_HIFLOAT4);

    OP_LOGI(context->GetNodeName(), "End to do InferShape4PartialRotaryMulQuant");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4PartialRotaryMulQuant(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "InferDtype4PartialRotaryMulQuant enter");
    context->SetOutputDataType(INDEX_OUTPUT_Y, ge::DT_HIFLOAT4);
    context->SetOutputDataType(INDEX_OUTPUT_Y_SCALE, ge::DT_HIFLOAT4_SCALE);
    OP_LOGI(context->GetNodeName(), "InferDtype4PartialRotaryMulQuant end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PartialRotaryMulQuant)
    .InferShape(InferShape4PartialRotaryMulQuant)
    .InferDataType(InferDtype4PartialRotaryMulQuant);
} // namespace ops

#endif
