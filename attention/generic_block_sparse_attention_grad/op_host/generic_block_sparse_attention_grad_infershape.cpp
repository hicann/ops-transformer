/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {

// IR input indices: query=0, key=1, value=2
static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t KEY_INDEX = 1;
static constexpr uint32_t VALUE_INDEX = 2;
static constexpr uint32_t DQ_OUT_INDEX = 0;
static constexpr uint32_t DK_OUT_INDEX = 1;
static constexpr uint32_t DV_OUT_INDEX = 2;

static ge::graphStatus InferShapeGenericBlockSparseAttentionGrad(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("GenericBlockSparseAttentionGrad", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    gert::Shape *dqOutShape = context->GetOutputShape(DQ_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dqOutShape);
    gert::Shape *dkOutShape = context->GetOutputShape(DK_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dkOutShape);
    gert::Shape *dvOutShape = context->GetOutputShape(DV_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dvOutShape);

    *dqOutShape = *queryShape;
    *dkOutShape = *keyShape;
    *dvOutShape = *valueShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GenericBlockSparseAttentionGrad).InferShape(InferShapeGenericBlockSparseAttentionGrad);

} // namespace ops
