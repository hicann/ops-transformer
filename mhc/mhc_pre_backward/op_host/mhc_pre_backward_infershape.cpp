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
 * \file mhc_pre_backward_infershape.cpp
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
using namespace ge;

namespace ops {

const constexpr int64_t PHI_INDEX = 1;
const constexpr int64_t GRAD_H_IN_INDEX = 3;
const constexpr int64_t GRAD_H_POST_INDEX = 4;
const constexpr int64_t GRAD_H_RES_INDEX = 5;

const constexpr int64_t OUT_GRAD_X_INDEX = 0;
const constexpr int64_t OUT_GRAD_PHI_INDEX = 1;
const constexpr int64_t OUT_GRAD_ALPHA_INDEX = 2;
const constexpr int64_t OUT_GRAD_BIAS_INDEX = 3;
const constexpr int64_t OUT_GRAD_GAMMA_INDEX = 4;

const constexpr int64_t BSD_DIM_NUM = 3;
const constexpr int64_t BSNN_DIM_NUM = 4;
const constexpr int64_t TD_DIM_NUM = 2;
const constexpr int64_t TN_DIM_NUM = 2;
const constexpr int64_t TNN_DIM_NUM = 3;
const constexpr int64_t TND_DIM_NUM = 3;
const constexpr int64_t GRAD_ALPHA_DIM_SIZE = 3;

const constexpr int64_t INDEX_0 = 0;
const constexpr int64_t INDEX_1 = 1;
const constexpr int64_t INDEX_2 = 2;

const constexpr int64_t INDEX_T = 0;
const constexpr int64_t INDEX_N = 1;
const constexpr int64_t INDEX_D_TND = 1;

ge::graphStatus GetInputShapes(InferShapeContext *context, const gert::Shape *&gradHInShape,
                               const gert::Shape *&gradHPostShape, const gert::Shape *&gradHResShape)
{
    gradHInShape = context->GetDynamicInputShape(GRAD_H_IN_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHInShape);
    gradHPostShape = context->GetDynamicInputShape(GRAD_H_POST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHPostShape);
    gradHResShape = context->GetDynamicInputShape(GRAD_H_RES_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHResShape);
    return GRAPH_SUCCESS;
}

ge::graphStatus ValidateInputDims(int64_t gradHInDimNum, int64_t gradHPostDimNum, int64_t gradHResDimNum)
{
    // Valid combinations:
    //   BSND: gradHIn=3D, gradHPost=3D, gradHRes=4D(BSNN) or 3D(BSN!)
    //   TND:  gradHIn=2D, gradHPost=2D, gradHRes=3D(TNN)  or 2D(TN!)
    if (gradHInDimNum != gradHPostDimNum) {
        return GRAPH_FAILED;
    }
    if (gradHInDimNum == BSD_DIM_NUM) {
        if (gradHResDimNum != BSNN_DIM_NUM && gradHResDimNum != TNN_DIM_NUM) {
            return GRAPH_FAILED;
        }
    } else if (gradHInDimNum == TD_DIM_NUM) {
        if (gradHResDimNum != TNN_DIM_NUM && gradHResDimNum != TN_DIM_NUM) {
            return GRAPH_FAILED;
        }
    } else {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferBSDFormat(const gert::Shape *gradHInShape, const gert::Shape *gradHPostShape,
                               gert::Shape *gradXShape, uint64_t &numsResidual, uint64_t &dimen)
{
    uint64_t batch = gradHInShape->GetDim(INDEX_0);
    uint64_t sequence = gradHInShape->GetDim(INDEX_1);
    dimen = gradHInShape->GetDim(INDEX_2);
    numsResidual = gradHPostShape->GetDim(INDEX_2);

    gradXShape->SetDimNum(BSNN_DIM_NUM);
    gradXShape->SetDim(0, batch);
    gradXShape->SetDim(1, sequence);
    gradXShape->SetDim(2, numsResidual);
    gradXShape->SetDim(3, dimen);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferTNDFormat(const gert::Shape *gradHInShape, const gert::Shape *gradHPostShape,
                               gert::Shape *gradXShape, uint64_t &numsResidual, uint64_t &dimen)
{
    uint64_t t = gradHInShape->GetDim(INDEX_T);
    dimen = gradHInShape->GetDim(INDEX_D_TND);
    numsResidual = gradHPostShape->GetDim(INDEX_N);

    gradXShape->SetDimNum(TND_DIM_NUM);
    gradXShape->SetDim(0, t);
    gradXShape->SetDim(1, numsResidual);
    gradXShape->SetDim(2, dimen);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferOutputShapes(gert::Shape *gradPhiShape, gert::Shape *gradAlphaShape, gert::Shape *gradBiasShape,
                                  gert::Shape *gradGammaShape, uint64_t numsResidual, uint64_t dimen,
                                  uint64_t fusionSize)
{
    gradPhiShape->SetDimNum(2);
    gradPhiShape->SetDim(0, fusionSize);
    gradPhiShape->SetDim(1, numsResidual * dimen);

    gradAlphaShape->SetDimNum(1);
    gradAlphaShape->SetDim(0, GRAD_ALPHA_DIM_SIZE);

    gradBiasShape->SetDimNum(1);
    gradBiasShape->SetDim(0, fusionSize);

    if (gradGammaShape != nullptr) {
        gradGammaShape->SetDimNum(2);
        gradGammaShape->SetDim(0, numsResidual);
        gradGammaShape->SetDim(1, dimen);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4mHCPreGrad(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape MhcPreBackward");
    const gert::Shape *gradHInShape = nullptr;
    const gert::Shape *gradHPostShape = nullptr;
    const gert::Shape *gradHResShape = nullptr;

    auto ret = GetInputShapes(context, gradHInShape, gradHPostShape, gradHResShape);
    if (ret != GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "input shapes are invalid");
        return ret;
    }

    auto gradXShape = context->GetOutputShape(OUT_GRAD_X_INDEX);
    auto gradPhiShape = context->GetOutputShape(OUT_GRAD_PHI_INDEX);
    auto gradAlphaShape = context->GetOutputShape(OUT_GRAD_ALPHA_INDEX);
    auto gradBiasShape = context->GetOutputShape(OUT_GRAD_BIAS_INDEX);
    auto gradGammaShape = context->GetOutputShape(OUT_GRAD_GAMMA_INDEX);

    auto gradHInDimNum = gradHInShape->GetDimNum();
    auto gradHPostDimNum = gradHPostShape->GetDimNum();
    auto gradHResDimNum = gradHResShape->GetDimNum();

    ret = ValidateInputDims(gradHInDimNum, gradHPostDimNum, gradHResDimNum);
    if (ret != GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "input dims invalid for MhcPreBackward");
        return ret;
    }

    uint64_t numsResidual = 0;
    uint64_t dimen = 0;

    if (gradHInDimNum == BSD_DIM_NUM) {
        ret = InferBSDFormat(gradHInShape, gradHPostShape, gradXShape, numsResidual, dimen);
    } else if (gradHInDimNum == TD_DIM_NUM) {
        ret = InferTNDFormat(gradHInShape, gradHPostShape, gradXShape, numsResidual, dimen);
    }
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    // Derive fusionSize from phi input shape (dim 0), which is platform-dependent:
    // A2 (ascend910b): N! + 2N, A3/A5 (ascend950): N^2 + 2N
    auto phiShape = context->GetInputShape(PHI_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, phiShape);
    uint64_t fusionSize = phiShape->GetDim(INDEX_0);

    ret =
        InferOutputShapes(gradPhiShape, gradAlphaShape, gradBiasShape, gradGammaShape, numsResidual, dimen, fusionSize);
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape MhcPreBackward");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4mHCPreGrad(gert::InferDataTypeContext *context)
{
    auto xGradType = context->GetInputDataType(GRAD_H_IN_INDEX);
    context->SetOutputDataType(OUT_GRAD_X_INDEX, xGradType);
    context->SetOutputDataType(OUT_GRAD_PHI_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_GRAD_ALPHA_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_GRAD_BIAS_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_GRAD_GAMMA_INDEX, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPreBackward).InferShape(InferShape4mHCPreGrad).InferDataType(InferDataType4mHCPreGrad);
} // namespace ops
