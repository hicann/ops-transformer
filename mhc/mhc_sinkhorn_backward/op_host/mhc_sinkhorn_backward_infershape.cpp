/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_sinkhorn_backward_infershape.cpp
 * \brief mhc_sinkhorn_backward_infershape
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"
#include "util/shape_util.h"

using namespace ge;
using namespace std;

namespace ops {
const static int64_t GRAD_Y_INPUT_INDEX = 0;
const static int64_t GRAD_INPUT_OUTPUT_INDEX = 0;

static constexpr size_t BSNN_DIMS = 4;
static constexpr size_t TNN_DIMS = 3;

static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2LL;

static ge::graphStatus InferShape4MhcSinkhornBackward(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do MhcSinkhornBackwardInfershape.");
    const gert::Shape* gradYShape = context->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYShape);
    gert::Shape* gradInputShape = context->GetOutputShape(GRAD_INPUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShape);

    if (Ops::Base::IsUnknownRank(*gradYShape)) {
        gradInputShape->SetDimNum(0);
        gradInputShape->AppendDim(UNKNOWN_RANK_DIM_VALUE);
        OP_LOGD(context->GetNodeName(), "MhcSinkhornBackward infershape handles unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    size_t gradYDims = gradYShape->GetDimNum();
    OP_CHECK_IF(
        (gradYDims != TNN_DIMS) && (gradYDims != BSNN_DIMS),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "gradY", std::to_string(gradYDims).c_str(),
                                                 "the dim of gradY must be 3 or 4"),
        return ge::GRAPH_FAILED);

    gradInputShape->SetDimNum(gradYDims);
    for (size_t i = 0; i < gradYDims; ++i) {
        gradInputShape->SetDim(i, gradYShape->GetDim(i));
    }
    OP_LOGD(context->GetNodeName(), "End to do MhcSinkhornBackwardInfershape.");
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4MhcSinkhornBackward(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "MhcSinkhornBackwardInferDtype enter");
    const ge::DataType gradYDType = context->GetInputDataType(GRAD_Y_INPUT_INDEX);
    context->SetOutputDataType(GRAD_INPUT_OUTPUT_INDEX, gradYDType);
    OP_LOGD(context->GetNodeName(), "MhcSinkhornBackwardInferDtype end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcSinkhornBackward)
    .InferShape(InferShape4MhcSinkhornBackward)
    .InferDataType(InferDtype4MhcSinkhornBackward);
} // namespace ops
