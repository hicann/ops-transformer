/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mla_prolog_v2_infershape.h"

using namespace ge;

namespace ops {

ge::graphStatus SetMlaPrologV2ShapeDim(const MlaPrologProtoShapeParam& shapeParam, gert::InferShapeContext* context)
{
    auto apiRet = SetMlaPrologShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "SetMlaPrologShapeDim failed"),
                return ge::GRAPH_FAILED);

    // set output shape
    auto dequantScaleXShape = context->GetOptionalInputShape(DEQUANT_SCALE_X_INDEX);
    auto quantScaleCkvShape = context->GetOptionalInputShape(QUANT_SCALE_CKV_INDEX);
    bool isQuantCkv = (dequantScaleXShape != nullptr && quantScaleCkvShape != nullptr); // full quant

    // dequantScaleQNope: (B*S, N ,1) | (T, N, 1). (1) if not enabled
    auto dequantScaleQNopeShape = context->GetOutputShape(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dequantScaleQNopeShape);

    if (!isQuantCkv) {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, DIM_NUM_1);
    } else {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_3); // (B*S, N, 1) | (T, N, 1)
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_2, 1); // 1: Fix dim 1
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMlaPrologV2(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV2 infershape impl.");

    MlaPrologProtoShapeParam shapeParam{};
    auto apiRet = GetMlaPrologShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"),
                return ge::GRAPH_FAILED);

    apiRet = SetMlaPrologV2ShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context set output shape failed"),
                return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlaPrologV2(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV2 infershape impl.");

    OP_CHECK_IF((InferDataTypeMlaProlog(context) != GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "Context get input shape failed"), return ge::GRAPH_FAILED);

    // full quant
    bool isQuantCkv = (context->GetInputDataType(TOKEN_X_INDEX) == ge::DT_INT8 &&
                       context->GetOptionalInputDataType(QUANT_SCALE_CKV_INDEX) != ge::DT_UNDEFINED);

    context->SetOutputDataType(QUERY_INDEX, (isQuantCkv) ? ge::DT_INT8 : context->GetInputDataType(WEIGHT_UK_INDEX));
    context->SetOutputDataType(DEQUANT_SCALE_Q_NOPE_INDEX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaPrologV2).InferShape(InferShapeMlaPrologV2).InferDataType(InferDataTypeMlaPrologV2);
} // namespace ops
