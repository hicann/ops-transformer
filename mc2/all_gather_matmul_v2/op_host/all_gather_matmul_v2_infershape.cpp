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
 * \file all_gather_matmul_v2_infershape.cpp
 * \brief
 */
#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "mc2_hcom_topo_info.h"
#include "op_host/mc2_common_infershape.h"

using namespace ge;
namespace ops {

// input tensor index
const size_t INDEX_IN_X1 = 0;
// attr index
const size_t INDEX_ATTR_Y_DTYPE = 10;
// output tensor index
const size_t INDEX_OUT = 0;
const size_t INDEX_GATHER_OUT = 1;

static ge::graphStatus InferShapeAllGatherMatmulV2(gert::InferShapeContext *context)
{
    OP_LOGE_IF(AllGatherMatmulCommonInferShape(context, GATHER_OUT_V2) != GRAPH_SUCCESS, GRAPH_FAILED,
               context->GetNodeName(), "infer shape execute failed.");
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *isAmaxOut = attrs->GetAttrPointer<bool>(AG_IS_AMAX_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, isAmaxOut);
    gert::Shape *amaxOutShape = context->GetOutputShape(2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, amaxOutShape);
    if (*isAmaxOut) {
        amaxOutShape->SetDimNum(1);
        amaxOutShape->SetDim(0, 1);
    } else {
        amaxOutShape->SetDimNum(1);
        amaxOutShape->SetDim(0, 0);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAllGatherMatmulV2(gert::InferDataTypeContext *context)
{
    OP_LOGE_IF(InferMatmulOutputDataType(context, INDEX_IN_X1, INDEX_ATTR_Y_DTYPE, INDEX_OUT) != GRAPH_SUCCESS,
               GRAPH_FAILED, context->GetNodeName(), "Infer output dtype failed.");
    const auto x1Dtype = context->GetInputDataType(INDEX_IN_X1);
    context->SetOutputDataType(INDEX_GATHER_OUT, x1Dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AllGatherMatmulV2)
    .InferShape(InferShapeAllGatherMatmulV2)
    .InferDataType(InferDataTypeAllGatherMatmulV2);
} // namespace ops
