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
 * \file quant_grouped_matmul_inplace_add_infershape.cpp
 * \brief
 */
#include <algorithm>

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
const int64_t X1_INDEX = 0;
const int64_t X2_INDEX = 1;
const int64_t SCALE2_INDEX = 2;
const int64_t GROUPLIST_INDEX = 3;
const int64_t YREF_INDEX = 4;
const int64_t SCALE1_INDEX = 5;
constexpr const char* QUANT_GROUPED_MATMUL_INPLACE_ADD_OP_TYPE = "QuantGroupedMatmulInplaceAdd";

static ge::graphStatus IsInputTensorNull(const gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(X1_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(X2_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(SCALE2_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(GROUPLIST_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(YREF_INDEX));
    OP_CHECK_IF(
        context->GetOptionalInputShape(SCALE1_INDEX) == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QUANT_GROUPED_MATMUL_INPLACE_ADD_OP_TYPE, "scale1", "nullptr",
                                              "null check with context: context->GetOptionalInputShape(SCALE1_INDEX)"),
        return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4QuantGroupedMatmulInplaceAdd(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_CHECK_IF(IsInputTensorNull(context) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Some of the required inputs are null."), return GRAPH_FAILED);
    auto yRefShape = context->GetInputShape(YREF_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yRefShape);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *yRefShape;
    return GRAPH_SUCCESS;
}

// =========================================================================================

static graphStatus InferDataType4QuantGroupedMatmulInplaceAdd(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantGroupedMatmulInplaceAdd)
    .InferShape(InferShape4QuantGroupedMatmulInplaceAdd)
    .InferDataType(InferDataType4QuantGroupedMatmulInplaceAdd);
} // namespace ops
