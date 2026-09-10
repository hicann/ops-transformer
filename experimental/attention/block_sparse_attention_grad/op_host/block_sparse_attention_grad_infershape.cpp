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

static constexpr uint32_t QUERY_INDEX = 1;
static constexpr uint32_t KEY_INDEX = 2;
static constexpr uint32_t VALUE_INDEX = 3;
static constexpr uint32_t DQ_OUT_INDEX = 0;
static constexpr uint32_t DK_OUT_INDEX = 1;
static constexpr uint32_t DV_OUT_INDEX = 2;

static constexpr uint32_t ATTR_Q_INPUT_LAYOUT_INDEX = 0;
static constexpr uint32_t ATTR_KV_INPUT_LAYOUT_INDEX = 1;
static constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 2;

static constexpr uint32_t DIM_BSH = 3;
static constexpr uint32_t DIM_TND = 3;
static constexpr uint32_t DIM_BNSD = 4;

static constexpr uint32_t BSH_DIM_B = 0;
static constexpr uint32_t BSH_DIM_S = 1;
static constexpr uint32_t BSH_DIM_H = 2;

static constexpr uint32_t TND_DIM_T = 0;
static constexpr uint32_t TND_DIM_N = 1;
static constexpr uint32_t TND_DIM_D = 2;

static constexpr uint32_t BNSD_DIM_B = 0;
static constexpr uint32_t BNSD_DIM_N = 1;
static constexpr uint32_t BNSD_DIM_S = 2;
static constexpr uint32_t BNSD_DIM_D = 3;

static constexpr int32_t UNKNOWN_DIMS = -2;

static ge::graphStatus InferShapeBlockSparseAttentionGrad(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BlockSparseAttentionGrad", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Enter BlockSparseAttentionGrad InferShape impl.");

    // 获取Query shape
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    // 获取Key shape
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    // 获取Value shape
    const gert::Shape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    // 获取输出shape
    gert::Shape *dqOutShape = context->GetOutputShape(DQ_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dqOutShape);

    gert::Shape *dkOutShape = context->GetOutputShape(DK_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dkOutShape);

    gert::Shape *dvOutShape = context->GetOutputShape(DV_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dvOutShape);

    // 设置AttentionGradOut shape
    *dqOutShape = *queryShape;
    *dkOutShape = *keyShape;
    *dvOutShape = *valueShape;

    OP_LOGD(context->GetNodeName(), "BlockSparseAttentionGrad InferShape success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockSparseAttentionGrad).InferShape(InferShapeBlockSparseAttentionGrad);

} // namespace ops
