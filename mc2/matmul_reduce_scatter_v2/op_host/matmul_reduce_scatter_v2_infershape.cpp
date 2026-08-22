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
 * \file matmul_reduce_scatter_v2_infershape.cc
 * \brief
 */
#include "ops_utils.h"
#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "mc2_hcom_topo_info.h"
#include "op_host/mc2_common_infershape.h"

using namespace ge;
namespace ops {
const size_t Y_DTYPE = 9;

static ge::graphStatus InferShapeMatmulReduceScatterV2(gert::InferShapeContext *context)
{
    OP_LOGE_IF(InferMatmulReduceScatterCommon(context) != GRAPH_SUCCESS, GRAPH_FAILED, context->GetNodeName(),
               "infer shape execute failed.");
    const bool *isAmaxOut = context->GetAttrs()->GetAttrPointer<bool>(RS_IS_AMAX_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, isAmaxOut);
    gert::Shape *amaxOutShape = context->GetOutputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, amaxOutShape);
    if (*isAmaxOut) {
        amaxOutShape->SetDimNum(1);
        amaxOutShape->SetDim(0, 1);
    } else {
        amaxOutShape->SetDimNum(1);
        amaxOutShape->SetDim(0, 0);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMatmulReduceScatterV2(gert::InferDataTypeContext *context)
{
    // 如果是bf16/fp16 输入和输出保持一致，如果是fp8 则使用y_dtype
    auto d_type = ge::DataType::DT_FLOAT;
    if ((d_type == ge::DataType::DT_FLOAT16) || (d_type == ge::DataType::DT_BF16)) {
        d_type = context->GetInputDataType(0);
    } else {
        d_type = static_cast<ge::DataType>(*context->GetAttrs()->GetAttrPointer<int64_t>(Y_DTYPE));
    }
    context->SetOutputDataType(0, d_type);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulReduceScatterV2)
    .InferShape(InferShapeMatmulReduceScatterV2)
    .InferDataType(InferDataTypeMatmulReduceScatterV2);
} // namespace ops
