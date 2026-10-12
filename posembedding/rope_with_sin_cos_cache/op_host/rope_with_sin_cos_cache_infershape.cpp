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
 * \file rope_with_sin_cos_cache.cc
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

static constexpr int INPUT_NODE_NUM = 4;
static constexpr int OUTPUT_NODE_NUM = 2;

static constexpr int INDEX_INPUT_QUERY_IN = 1;
static constexpr int INDEX_INPUT_KEY_IN = 2;

static constexpr int INDEX_OUTPUT_QUERY_OUT = 0;
static constexpr int INDEX_OUTPUT_KEY_OUT = 1;

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForRopeWithSinCosCache(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("InferShape4RopeWithSinCosCache", "Context is nullptr, check failed.");
        return GRAPH_FAILED;
    }
    if (context->GetComputeNodeInputNum() != INPUT_NODE_NUM || context->GetComputeNodeOutputNum() != OUTPUT_NODE_NUM) {
        return GRAPH_FAILED;
    }

    const gert::Shape* query_in_shape = context->GetInputShape(INDEX_INPUT_QUERY_IN);
    OP_CHECK_NULL_WITH_CONTEXT(context, query_in_shape);
    const gert::Shape* key_in_shape = context->GetInputShape(INDEX_INPUT_KEY_IN);
    OP_CHECK_NULL_WITH_CONTEXT(context, key_in_shape);

    gert::Shape* query_out_shape = context->GetOutputShape(INDEX_OUTPUT_QUERY_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, query_out_shape);
    gert::Shape* key_out_shape = context->GetOutputShape(INDEX_OUTPUT_KEY_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, key_out_shape);

    *query_out_shape = *query_in_shape;
    *key_out_shape = *key_in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRopeWithSinCosCache(gert::InferDataTypeContext* context)
{
    const auto input_data_type_1 = context->GetInputDataType(1);
    context->SetOutputDataType(0, input_data_type_1);
    context->SetOutputDataType(1, input_data_type_1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RopeWithSinCosCache)
    .InferShape(InferShapeForRopeWithSinCosCache)
    .InferDataType(InferDataTypeForRopeWithSinCosCache);
} // namespace ops
