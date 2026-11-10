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
 * \file moe_distribute_combine_add_rms_norm_infershape.cpp
 * \brief
 */
#include "mc2_log.h"
#include "mc2_infer_failure_report.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr uint32_t IDX_ZERO = 0;
static constexpr uint32_t IDX_ONE = 1;
static constexpr uint32_t IDX_TWO = 2;
static constexpr uint32_t IDX_THREE = 3;
static constexpr uint32_t IDX_FIVE = 5;
static constexpr uint32_t IDX_EIGHT = 8;
static constexpr uint32_t IDX_THIRTEEN = 13;

static ge::graphStatus InferShapeCheck(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        Mc2Error::ReportInferFailure("MoeDistributeCombineAddRmsNorm",
                                     "InferShapeCheck: context actual=null; expected a materialized InferShapeContext");
        return GRAPH_FAILED;
    }

    static const char* OUTPUT_NAMES[] = {"y", "rstdOut", "x"};
    for (uint32_t i = 0; i < IDX_THREE; ++i) {
        if (context->GetOutputShape(i) == nullptr) {
            Mc2Error::ReportInferFailure(
                context->GetNodeName(),
                std::string("InferShapeCheck: output shape holder actual=null for ") + OUTPUT_NAMES[i]);
            return GRAPH_FAILED;
        }
    }

    static const char* INPUT_NAMES[] = {"expand_x",       "expert_ids",    "assist_info_for_combine",
                                        "ep_send_counts", "expert_scales", "residual_x",
                                        "gamma",          "tp_send_counts"};
    for (uint32_t i = 0; i < IDX_EIGHT - 1; ++i) {
        if (context->GetInputShape(i) == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), INPUT_NAMES[i]);
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineAddRmsNormInferShape(gert::InferShapeContext* context)
{
    if (InferShapeCheck(context) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do Infershape of MoeDistributeCombineAddRmsNormInferShape.");
    const gert::Shape* residualX = context->GetInputShape(IDX_FIVE);
    gert::Shape* x1OutShape = context->GetOutputShape(IDX_ZERO);
    gert::Shape* rstdOutShape = context->GetOutputShape(IDX_ONE);
    gert::Shape* x2OutShape = context->GetOutputShape(IDX_TWO);

    size_t dimNum = residualX->GetDimNum();
    OP_CHECK_IF(
        dimNum == 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "residual_x", std::to_string(dimNum).c_str(),
                                                 "rank must be positive before accessing the last dimension"),
        return GRAPH_FAILED);
    x1OutShape->SetDimNum(IDX_THREE);
    x1OutShape->SetDim(IDX_ZERO, residualX->GetDim(IDX_ZERO));
    x1OutShape->SetDim(IDX_ONE, IDX_ONE);
    x1OutShape->SetDim(IDX_TWO, residualX->GetDim(dimNum - 1));
    rstdOutShape->SetDimNum(IDX_THREE);
    rstdOutShape->SetDim(IDX_ZERO, residualX->GetDim(IDX_ZERO));
    rstdOutShape->SetDim(IDX_ONE, IDX_ONE);
    rstdOutShape->SetDim(IDX_TWO, IDX_ONE);
    x2OutShape->SetDimNum(IDX_THREE);
    x2OutShape->SetDim(IDX_ZERO, residualX->GetDim(IDX_ZERO));
    x2OutShape->SetDim(IDX_ONE, IDX_ONE);
    x2OutShape->SetDim(IDX_TWO, residualX->GetDim(dimNum - 1));

    OP_LOGD(context->GetNodeName(), "End to do MoeDistributeCombineAddRmsNormInferShape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineAddRmsNormInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "MoeDistributeCombineAddRmsNormInferDataType begin");
    context->SetOutputDataType(IDX_ZERO, ge::DT_BF16);
    context->SetOutputDataType(IDX_ONE, ge::DT_FLOAT);
    context->SetOutputDataType(IDX_TWO, ge::DT_BF16);
    OP_LOGD(context->GetNodeName(), "MoeDistributeCombineAddRmsNormInferDataType end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeCombineAddRmsNorm)
    .InferShape(MoeDistributeCombineAddRmsNormInferShape)
    .InferDataType(MoeDistributeCombineAddRmsNormInferDataType);
} // namespace ops
