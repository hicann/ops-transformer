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
 * \file apply_rotary_pos_emb_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "infershape_broadcast_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT0 = 0;
static constexpr size_t INPUT1 = 1;
static constexpr size_t INPUT2 = 2;
static constexpr size_t INPUT3 = 3;
static constexpr size_t OUTPUT_0_IDX = 0;
static constexpr size_t OUTPUT_1_IDX = 1;
static constexpr int64_t NEG_TWO = -2;

static bool BroadcastDim(int64_t& dim1, const int64_t dim2)
{
    /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
    dim -1  0  1  d2
    -1  -1  0 -1  d2
    0    0  0  0  E
    1   -1  0  1  d2
    d1  d1  E  d1 E
    */

    // no need to broadcast
    if (dim1 == dim2) {
        return true;
    }
    if ((dim1 != 1) && (dim2 != 1)) {
        // dynamic shape infershape
        if ((dim1 == -1) || (dim2 == -1)) {
            dim1 = (dim1 == -1) ? dim2 : dim1;
            return true;
        }

        // 部分旋转位置编码情况
        if (dim1 > dim2) {
            return true;
        }

        OP_LOGE("BroadcastDim", "dim1 %ld and dim2 %ld cannot broadcast!", dim1, dim2);
        return false;
    }

    // static shape infershape
    dim1 = (dim1 == 1) ? dim2 : dim1;

    return true;
}

/*
 * @brief: broadcast new shape to output shape
 * @param [in] shape: const gert::Shape*, new shape to broadcast
 * @param [in/out] shape_output: gert::Shape*, output shape
 * @return succeed or not
 */
static bool BroadcastShapeToOutShape(const gert::Shape* shape, gert::Shape* shape_output)
{
    OP_LOGD("BroadcastShapeToOutShape", "start broadcast %s to %s!", Ops::Base::ToString(*shape).c_str(),
            Ops::Base::ToString(*shape_output).c_str());

    if (Ops::Base::IsUnknownRank(*shape) || Ops::Base::IsUnknownRank(*shape_output)) {
        OP_LOGD("BroadcastShapeToOutShape", "the input shape is [-2], set output shape is [-2]!");
        shape_output->SetDimNum(1);
        shape_output->SetDim(0, NEG_TWO);
        return true;
    }

    size_t shape_len = shape->GetDimNum();
    size_t shape_y_len = shape_output->GetDimNum();
    if (shape_len > shape_y_len) {
        shape_output->SetDimNum(shape_len);
        size_t len_sub = shape_len - shape_y_len;
        for (size_t i = shape_y_len; i > 0; i--) {
            int64_t dim1 = shape->GetDim(len_sub + i - 1);
            int64_t dim2 = shape_output->GetDim(i - 1);
            if (!BroadcastDim(dim1, dim2)) {
                OP_LOGE("BroadcastShapeToOutShape", "dim1 %ld and dim2 %ld cannot broadcast!", dim1, dim2);
                return false;
            }
            shape_output->SetDim(len_sub + i - 1, dim1);
        }
        for (size_t i = 0; i < len_sub; i++) {
            shape_output->SetDim(i, shape->GetDim(i));
        }
    } else {
        auto len_sub = shape_y_len - shape_len;
        for (size_t i = 0; i < shape_len; i++) {
            int64_t dim1 = shape_output->GetDim(len_sub + i);
            int64_t dim2 = shape->GetDim(i);
            if (!BroadcastDim(dim1, dim2)) {
                OP_LOGE("BroadcastShapeToOutShape", "dim1 %ld and dim2 %ld cannot broadcast!", dim1, dim2);
                return false;
            }
            shape_output->SetDim(len_sub + i, dim1);
        }
    }
    return true;
}

static bool BroadcastShape(const gert::Shape* in1_shape, const gert::Shape* in2_shape, gert::Shape* out_shape)
{
    *out_shape = *in1_shape;

    OP_CHECK_IF(!BroadcastShapeToOutShape(in2_shape, out_shape),
                OP_LOGE("BroadcastShape", "shape [%s] and [%s] can not broadcast.",
                        Ops::Base::ToString(*in1_shape).c_str(), Ops::Base::ToString(*in2_shape).c_str()),
                return false);
    return true;
}

static ge::graphStatus ApplyRotaryPosEmbInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do ApplyRotaryPosEmbInferShape");
    // get input shapes
    const gert::Shape* qShape = context->GetInputShape(INPUT0);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const gert::Shape* kShape = context->GetInputShape(INPUT1);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    const gert::Shape* cosShape = context->GetInputShape(INPUT2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosShape);
    const gert::Shape* sinShape = context->GetInputShape(INPUT3);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinShape);
    // get output shapes
    gert::Shape* qOutshape = context->GetOutputShape(OUTPUT_0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutshape);
    gert::Shape* kOutshape = context->GetOutputShape(OUTPUT_1_IDX);
    if (Ops::Base::IsUnknownRank(*qShape)) {
        qOutshape->SetDimNum(1);
        qOutshape->SetDim(0, NEG_TWO);
    } else {
        if (!Ops::Base::IsUnknownRank(*cosShape)) {
            OP_CHECK_IF(!BroadcastShape(qShape, cosShape, qOutshape),
                        OP_LOGE(context->GetNodeName(), "cosShape [%s] and qShape [%s] can not broadcast.",
                                Ops::Base::ToString(*cosShape).c_str(), Ops::Base::ToString(*qShape).c_str()),
                        return ge::GRAPH_FAILED);
        } else if (!Ops::Base::IsUnknownRank(*sinShape)) {
            OP_CHECK_IF(!BroadcastShape(qShape, sinShape, qOutshape),
                        OP_LOGE(context->GetNodeName(), "sinShape [%s] and qShape [%s] can not broadcast.",
                                Ops::Base::ToString(*sinShape).c_str(), Ops::Base::ToString(*qShape).c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            *qOutshape = *qShape;
        }
    }

    if (Ops::Base::IsUnknownRank(*kShape)) {
        kOutshape->SetDimNum(1);
        kOutshape->SetDim(0, NEG_TWO);
    } else {
        if (!Ops::Base::IsUnknownRank(*cosShape)) {
            OP_CHECK_IF(!BroadcastShape(kShape, cosShape, kOutshape),
                        OP_LOGE(context->GetNodeName(), "cosShape [%s] and kShape [%s] can not broadcast.",
                                Ops::Base::ToString(*cosShape).c_str(), Ops::Base::ToString(*kShape).c_str()),
                        return ge::GRAPH_FAILED);
        } else if (!Ops::Base::IsUnknownRank(*sinShape)) {
            OP_CHECK_IF(!BroadcastShape(kShape, sinShape, kOutshape),
                        OP_LOGE(context->GetNodeName(), "sinShape [%s] and kShape [%s] can not broadcast.",
                                Ops::Base::ToString(*sinShape).c_str(), Ops::Base::ToString(*kShape).c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            *kOutshape = *kShape;
        }
    }
    OP_LOGD(context->GetNodeName(), "End to do ApplyRotaryPosEmbInferShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyRotaryPosEmb).InferShape(ApplyRotaryPosEmbInferShape);
} // namespace ops
