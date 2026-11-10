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
#include "mc2_infer_failure_report.h"
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

static ge::graphStatus InferShapeAllGatherMatmulV2(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                Mc2Error::ReportInferFailure("AllGatherMatmulV2",
                                             "InferShape entry: context actual=null; expected materialized context"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AllGatherMatmulCommonInferShape(context, GATHER_OUT_V2) != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "infer shape execute failed."), return GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                Mc2Error::ReportInferFailure(
                    context->GetNodeName(),
                    "InferShapeAllGatherMatmulV2: GetAttrs actual=null; expected schema attribute metadata"),
                return ge::GRAPH_FAILED);
    const bool* isAmaxOut = attrs->GetAttrPointer<bool>(AG_IS_AMAX_OUT);
    OP_CHECK_IF(isAmaxOut == nullptr,
                Mc2Error::ReportInferFailure(context->GetNodeName(),
                                             "InferShapeAllGatherMatmulV2: is_amax_out pointer actual=null; expected "
                                             "materialized schema-defaulted attribute"),
                return ge::GRAPH_FAILED);
    gert::Shape* amaxOutShape = context->GetOutputShape(2);
    OP_CHECK_IF(
        amaxOutShape == nullptr,
        Mc2Error::ReportInferFailure(context->GetNodeName(), "InferShapeAllGatherMatmulV2: Output amax_out shape "
                                                             "holder actual=null; expected materialized output holder"),
        return ge::GRAPH_FAILED);
    if (*isAmaxOut) {
        amaxOutShape->SetDimNum(1);
        amaxOutShape->SetDim(0, 1);
    } else {
        amaxOutShape->SetDimNum(1);
        amaxOutShape->SetDim(0, 0);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAllGatherMatmulV2(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(InferMatmulOutputDataType(context, INDEX_IN_X1, INDEX_ATTR_Y_DTYPE, INDEX_OUT) != GRAPH_SUCCESS,
                OP_LOGE_LIBOPAPI_REPORT(context->GetNodeName(), "Infer output dtype failed."), return GRAPH_FAILED);
    const auto x1Dtype = context->GetInputDataType(INDEX_IN_X1);
    context->SetOutputDataType(INDEX_GATHER_OUT, x1Dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AllGatherMatmulV2)
    .InferShape(InferShapeAllGatherMatmulV2)
    .InferDataType(InferDataTypeAllGatherMatmulV2);
} // namespace ops
