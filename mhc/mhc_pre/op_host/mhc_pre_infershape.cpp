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
 * \file mhc_pre_infershape.cpp
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

// Input tensor indices.
const constexpr int64_t X_INDEX = 0;
const constexpr int64_t PHI_INDEX = 1;
const constexpr int64_t ALPHA_INDEX = 2;

// Output tensor indices.
const constexpr int64_t OUT_H_IN_INDEX = 0;
const constexpr int64_t OUT_H_POST_INDEX = 1;
const constexpr int64_t OUT_H_RES_INDEX = 2;
const constexpr int64_t OUT_INV_RMS_INDEX = 3;
const constexpr int64_t OUT_H_MIX_INDEX = 4;
const constexpr int64_t OUT_H_PRE_INDEX = 5;

// Supported input ranks and dynamic-shape marker.
const constexpr int64_t BSND_DIM_NUM = 4;
const constexpr int64_t TND_DIM_NUM = 3;
const constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

static void SetShape3D(gert::Shape *shape, uint64_t d0, uint64_t d1, uint64_t d2)
{
    shape->SetDimNum(3);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
}

static void SetShape2D(gert::Shape *shape, uint64_t d0, uint64_t d1)
{
    shape->SetDimNum(2);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
}

static void SetShape4D(gert::Shape *shape, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
{
    shape->SetDimNum(4);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
    shape->SetDim(3, d3);
}

static void SetShape1D(gert::Shape *shape, uint64_t d0)
{
    shape->SetDimNum(1);
    shape->SetDim(0, d0);
}

static void SetShapeFromX(gert::Shape *dst, const gert::Shape *src)
{
    dst->SetDimNum(src->GetDimNum());
    for (int64_t i = 0; i < src->GetDimNum(); ++i) {
        dst->SetDim(i, src->GetDim(i));
    }
}

static uint64_t Factorial(uint64_t n)
{
    uint64_t r = 1;
    for (uint64_t i = 2; i <= n; ++i) {
        r *= i;
    }
    return r;
}

static ge::graphStatus InferShape4MhcPre(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape MhcPre");
    const gert::Shape *xShape = context->GetDynamicInputShape(X_INDEX, 0);
    const gert::Shape *phiShape = context->GetDynamicInputShape(PHI_INDEX, 0);
    const gert::Shape *alphaShape = context->GetDynamicInputShape(ALPHA_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, phiShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, alphaShape);

    int64_t phiDim = phiShape->GetDimNum();
    int64_t xDim = xShape->GetDimNum();
    OP_CHECK_IF(phiDim != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "phi", std::to_string(phiDim).c_str(), "2"),
                return GRAPH_FAILED);
    OP_CHECK_IF(xDim != BSND_DIM_NUM && xDim != TND_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDim).c_str(), "3 or 4"),
                return GRAPH_FAILED);

    int64_t alphaDim = alphaShape->GetDimNum();
    OP_CHECK_IF(alphaDim != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "alpha", std::to_string(alphaDim).c_str(), "1"),
                return GRAPH_FAILED);

    int64_t alphaDimSize = alphaShape->GetDim(0);
    OP_CHECK_IF(alphaDimSize != 2 && alphaDimSize != 3,
                OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "alpha", Ops::Base::ToString(*alphaShape).c_str(),
                                          "[2] or [3]"),
                return GRAPH_FAILED);
    // has residual
    bool hasResi = (alphaDimSize == 3);

    gert::Shape *outShapes[6] = {context->GetOutputShape(OUT_H_IN_INDEX),  context->GetOutputShape(OUT_H_POST_INDEX),
                                 context->GetOutputShape(OUT_H_RES_INDEX), context->GetOutputShape(OUT_INV_RMS_INDEX),
                                 context->GetOutputShape(OUT_H_MIX_INDEX), context->GetOutputShape(OUT_H_PRE_INDEX)};

    uint64_t phiDim0 = phiShape->GetDim(0);
    uint64_t n = 0;
    uint64_t d = 0;
    uint64_t nn = 0;
    bool isFac = false;
    if (xDim == BSND_DIM_NUM) {
        n = xShape->GetDim(2);
        d = xShape->GetDim(3);
    } else {
        n = xShape->GetDim(1);
        d = xShape->GetDim(2);
    }

    uint64_t nn_fac = Factorial(n);
    uint64_t nn_pow = n * n;
    uint64_t doubleN = 2 * n;
    OP_CHECK_IF(
        !hasResi && phiDim0 != doubleN,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "phi", Ops::Base::ToString(*phiShape).c_str(),
                                              "alpha shape [2] requires phi shape [2n, nD]"),
        return GRAPH_FAILED);
    OP_CHECK_IF(
        hasResi && phiDim0 != nn_fac + doubleN && phiDim0 != nn_pow + doubleN,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "phi", Ops::Base::ToString(*phiShape).c_str(),
                                              "alpha shape [3] requires phi to contain the residual branch"),
        return GRAPH_FAILED);
    if (phiDim0 == (nn_fac + doubleN)) {
        nn = nn_fac;
        isFac = true;
    } else {
        nn = nn_pow;
    }

    if (xDim == BSND_DIM_NUM) {
        // b,s,n,d
        uint64_t b = xShape->GetDim(0);
        uint64_t s = xShape->GetDim(1);
        // dim of phi,bias
        uint64_t matN = hasResi ? (nn + 2 * n) : (2 * n);
        SetShape3D(outShapes[OUT_H_IN_INDEX], b, s, d);
        SetShape3D(outShapes[OUT_H_POST_INDEX], b, s, n);
        if (hasResi && isFac) {
            SetShape3D(outShapes[OUT_H_RES_INDEX], b, s, nn);
        } else if (hasResi) {
            SetShape4D(outShapes[OUT_H_RES_INDEX], b, s, n, n);
        } else {
            SetShape1D(outShapes[OUT_H_RES_INDEX], 0);
        }
        SetShape2D(outShapes[OUT_INV_RMS_INDEX], b, s);
        SetShape3D(outShapes[OUT_H_MIX_INDEX], b, s, matN);
        SetShape3D(outShapes[OUT_H_PRE_INDEX], b, s, n);
    } else {
        // t,n,d
        uint64_t t = xShape->GetDim(0);
        // dim of phi,bias
        uint64_t matN = hasResi ? (nn + 2 * n) : (2 * n);
        SetShape2D(outShapes[OUT_H_IN_INDEX], t, d);
        SetShape2D(outShapes[OUT_H_POST_INDEX], t, n);
        if (hasResi && isFac) {
            SetShape2D(outShapes[OUT_H_RES_INDEX], t, nn);
        } else if (hasResi) {
            SetShape3D(outShapes[OUT_H_RES_INDEX], t, n, n);
        } else {
            SetShape1D(outShapes[OUT_H_RES_INDEX], 0);
        }
        SetShape1D(outShapes[OUT_INV_RMS_INDEX], t);
        SetShape2D(outShapes[OUT_H_MIX_INDEX], t, matN);
        SetShape2D(outShapes[OUT_H_PRE_INDEX], t, n);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape MhcPre");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4MhcPre(gert::InferDataTypeContext *context)
{
    const auto xDtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(OUT_H_IN_INDEX, xDtype);
    context->SetOutputDataType(OUT_H_POST_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_H_RES_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_INV_RMS_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_H_MIX_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_H_PRE_INDEX, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPre).InferShape(InferShape4MhcPre).InferDataType(InferDataType4MhcPre);
} // namespace ops
