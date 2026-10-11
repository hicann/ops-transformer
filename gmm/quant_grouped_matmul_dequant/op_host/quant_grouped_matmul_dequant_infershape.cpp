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
 * \file quant_grouped_matmul_dequant_infershape.cpp
 * \brief
 */

#include "runtime_util.h"
using namespace ge;
namespace ops {
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t WEIGHT_SCALE_INPUT_IDX = 2;
static constexpr int64_t Y_OUTPUT_IDX = 0;

static ge::graphStatus QuantMatmulDequantInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* weightScaleShape = context->GetInputShape(WEIGHT_SCALE_INPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightScaleShape);
    gert::Shape* yShape = context->GetOutputShape(Y_OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto M = xShape->GetDim(0);
    auto weightScaleDimNum = weightScaleShape->GetDimNum();
    auto N = weightScaleShape->GetDim(weightScaleDimNum - 1);
    gert::Shape shape{M, N};
    *yShape = shape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus QuantMatmulDequantInferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType xDtype = context->GetInputDataType(X_INPUT_IDX);
    context->SetOutputDataType(Y_OUTPUT_IDX, xDtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantMatmulDequant)
    .InferShape(QuantMatmulDequantInferShape)
    .InferDataType(QuantMatmulDequantInferDataType);

IMPL_OP_INFERSHAPE(QuantGroupedMatmulDequant)
    .InferShape(QuantMatmulDequantInferShape)
    .InferDataType(QuantMatmulDequantInferDataType);
} // namespace ops
