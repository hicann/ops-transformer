/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_fused_topk_infershape.cpp
 * \brief
 */

#include <string>
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr int64_t SIZE_1 = 1;
static constexpr int64_t SIZE_2 = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
static constexpr int64_t X_INDEX = 0;
static constexpr int64_t ADD_NUM_INDEX = 1;
static constexpr int64_t Y_INDEX = 0;
static constexpr int64_t INDICES_INDEX = 1;
static constexpr int64_t ATTR_TOP_K_INDEX = 3;

static ge::graphStatus CheckAddNumShape(gert::InferShapeContext* context, const gert::Shape* addNumShape,
                                        const int64_t expertNum)
{
    const int64_t addNumDimNum = static_cast<int64_t>(addNumShape->GetDimNum());
    if (addNumDimNum == SIZE_1 && addNumShape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_) {
        return ge::GRAPH_SUCCESS;
    }
    if (addNumDimNum != SIZE_1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "add_num", std::to_string(addNumDimNum), "1 or dynamic");
        return ge::GRAPH_FAILED;
    }
    const int64_t addNumDim = addNumShape->GetDim(0);
    if (addNumDim < NEG_ONE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "add_num", Ops::Base::ToString(*addNumShape),
                                              "dim of add_num must be greater than or equal to -1");
        return ge::GRAPH_FAILED;
    }
    if (addNumDim != NEG_ONE && expertNum != NEG_ONE && addNumDim != expertNum) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "add_num", Ops::Base::ToString(*addNumShape),
                                              "dim of add_num must be equal to expert num of x");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDimsAndAttr(gert::InferShapeContext* context, const gert::Shape* xShape,
                                             const gert::Shape* addNumShape, const int64_t xDimNum, const int64_t topK)
{
    if (xDimNum == 1) {
        if (xShape->GetDim(0) != UNKNOWN_RANK_DIM_VALUE_) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape),
                                                  "dynamic dim of x should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (xDimNum != SIZE_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDimNum), "2 or dynamic");
        return ge::GRAPH_FAILED;
    }

    if (topK <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "top_k", std::to_string(topK),
                                              "topK must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (xDimNum == SIZE_2) {
        const int64_t numToken = xShape->GetDim(0);
        const int64_t expertNum = xShape->GetDim(1);
        if (numToken < NEG_ONE || expertNum < NEG_ONE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape),
                                                  "x rows and expert num must be greater than or equal to -1");
            return ge::GRAPH_FAILED;
        }
        if (expertNum != NEG_ONE && topK > expertNum) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "top_k", std::to_string(topK),
                                                  "topK must not be greater than expert num of x");
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t expertNumForAddNum = xDimNum == SIZE_2 ? xShape->GetDim(1) : NEG_ONE;
    if (CheckAddNumShape(context, addNumShape, expertNumForAddNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4MoeFusedTopk(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do MoeFusedTopkInfershape.");

    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* addNumShape = context->GetInputShape(ADD_NUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, addNumShape);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* indicesShape = context->GetOutputShape(INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* topKPtr = attrs->GetAttrPointer<int64_t>(ATTR_TOP_K_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topKPtr);
    const int64_t topK = *topKPtr;
    OP_LOGD(context, "x shape is: %s, topK is: %ld.", Ops::Base::ToString(*xShape).c_str(), topK);

    // 参数校验
    const int64_t xDimNum = static_cast<int64_t>(xShape->GetDimNum());
    if (CheckInputDimsAndAttr(context, xShape, addNumShape, xDimNum, topK) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 未知秩(-2)或未知维(-1)输入时行数保持未知(-1)，不读取/计算具体维度
    const int64_t numToken = xDimNum == 1 ? NEG_ONE : xShape->GetDim(0);

    yShape->SetDimNum(SIZE_2);
    yShape->SetDim(0U, numToken);
    yShape->SetDim(1U, topK);

    indicesShape->SetDimNum(SIZE_2);
    indicesShape->SetDim(0U, numToken);
    indicesShape->SetDim(1U, topK);

    OP_LOGD(context, "y shape is: %s after infershape.", Ops::Base::ToString(*yShape).c_str());
    OP_LOGD(context, "indices shape is: %s after infershape.", Ops::Base::ToString(*indicesShape).c_str());
    OP_LOGD(context, "End to do MoeFusedTopkInfershape.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeFusedTopk).InferShape(InferShape4MoeFusedTopk);
} // namespace ops
