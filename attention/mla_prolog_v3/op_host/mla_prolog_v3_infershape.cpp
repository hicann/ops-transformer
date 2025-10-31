/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mla_prolog_v3_infershape.h"

using namespace ge;

namespace ops {

ge::graphStatus SetMlaPrologV3ShapeDim(const MlaProlgoProtoShapeParam &shapeParam, gert::InferShapeContext* context)
{
    auto apiRet = SetMlaPrologShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "SetMlaPrologShapeDim failed"), return ge::GRAPH_FAILED);

    // set output shape
    auto dequantScaleXShape = context->GetOptionalInputShape(DEQUANT_SCALE_X_INDEX);
    auto quantScaleCkvShape = context->GetOptionalInputShape(QUANT_SCALE_CKV_INDEX);
    bool isQuantCkv = (dequantScaleXShape != nullptr && quantScaleCkvShape != nullptr);  // full quant

    // dequantScaleQNope: (B*S, N ,1) | (T, N, 1). (1) if not enabled
    auto dequantScaleQNopeShape = context->GetOutputShape(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dequantScaleQNopeShape);

    auto queryNormShape = context->GetOutputShape(QUERY_NORM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryNormShape);

    auto dequantScaleQNormShape = context->GetOutputShape(DEQUANT_SCALE_Q_NORM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dequantScaleQNormShape);

    if (!isQuantCkv) {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, DIM_NUM_1);
    } else {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_3);                   // (B*S, N, 1) | (T, N, 1)
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_2, 1);                 // 1: Fix dim 1
    }

    // 预留接口，不做shape推导
    queryNormShape->SetDimNum(DIM_NUM_1);
    queryNormShape->SetDim(DIM_INDEX_0, DIM_NUM_1);
    dequantScaleQNormShape->SetDimNum(DIM_NUM_1);
    dequantScaleQNormShape->SetDim(DIM_INDEX_0, DIM_NUM_1);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMlaPrologV3(gert::InferShapeContext* context) {
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV3 infershape impl.");

    MlaProlgoProtoShapeParam shapeParam {};
    auto apiRet = GetMlaPrologShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"), return ge::GRAPH_FAILED);

    apiRet = SetMlaPrologV3ShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context set output shape failed"), return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlaPrologV3(gert::InferDataTypeContext* context) {
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV3 infershape impl.");

    OP_CHECK_IF((InferDataTypeMlaProlog(context) != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"),
        return ge::GRAPH_FAILED);

    // full quant
    bool isQuantCkv = (context->GetInputDataType(TOKEN_X_INDEX) == ge::DT_INT8 &&
        context->GetOptionalInputDataType(QUANT_SCALE_CKV_INDEX) != ge::DT_UNDEFINED);

    context->SetOutputDataType(QUERY_INDEX, (isQuantCkv) ? ge::DT_INT8 : context->GetInputDataType(WEIGHT_UK_INDEX));
    context->SetOutputDataType(DEQUANT_SCALE_Q_NOPE_INDEX, ge::DT_FLOAT);

    // 预留接口，不做推导
    context->SetOutputDataType(QUERY_NORM_INDEX, context->GetInputDataType(WEIGHT_UQ_QR_INDEX));
    context->SetOutputDataType(DEQUANT_SCALE_Q_NORM_INDEX, ge::DT_FLOAT);

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaPrologV3).InferShape(InferShapeMlaPrologV3).InferDataType(InferDataTypeMlaPrologV3);
}  // namespace ops