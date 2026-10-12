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
 * \file ring_attention_update_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include "platform/platform_info.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "ring_attention_update_tiling.h"
#include "ring_attention_update_tiling_arch35.h"

namespace optiling {
using namespace Ops::Base;

constexpr uint32_t REGBASE_KEY = 100;
constexpr uint32_t SOFTMAX_TND_KEY = 10;
constexpr uint32_t TND_KEY = 20;
constexpr uint32_t DTYPE_KEY_FP16 = 0;
constexpr uint32_t DTYPE_KEY_BF16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;

constexpr uint32_t SIZE_B32 = 4;
constexpr uint32_t SIZE_B16 = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t REPEAT_NUM_B32 = 256 / 4;

constexpr size_t CONST_ZERO = 0;
constexpr size_t CONST_ONE = 1;
constexpr size_t CONST_TWO = 2;
constexpr size_t CONST_THREE = 3;
constexpr size_t CONST_FOUR = 4;

constexpr size_t SOFTMAX_TAIL = 8;

constexpr uint64_t BUFFER_NUM_IN_QUE = 2;

template <typename T>
inline T AlignUp(T num, T rnd)
{
    return rnd == 0 ? 0 : (num + rnd - 1) / rnd * rnd;
}

template <typename T>
inline T AlignDown(T num, T rnd)
{
    return rnd == 0 ? 0 : num / rnd * rnd;
}

template <typename T>
inline T CeilDiv(T num, T div)
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

static ge::graphStatus SafeDivisionCheck(int64_t inputNum)
{
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    } else {
        return ge::GRAPH_SUCCESS;
    }
}

static ge::graphStatus RingAttentionUpdateRegbaseCheckDtype(const gert::TilingContext* context)
{
    auto prevAttnTensor = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevAttnTensor);
    ge::DataType prevAttnDtype = prevAttnTensor->GetDataType();
    if (prevAttnDtype != ge::DT_FLOAT16 && prevAttnDtype != ge::DT_BF16 && prevAttnDtype != ge::DT_FLOAT) {
        std::string prevAttnDtypeStr = ToString(prevAttnDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "prev_attn_out", prevAttnDtypeStr.c_str(),
                                  "FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }

    auto prevSoftmaxMaxTensor = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxMaxTensor);
    ge::DataType prevSoftmaxMaxDtype = prevSoftmaxMaxTensor->GetDataType();
    if (prevSoftmaxMaxDtype != ge::DT_FLOAT) {
        std::string prevSoftmaxMaxDtypeStr = ToString(prevSoftmaxMaxDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "prev_softmax_max", prevSoftmaxMaxDtypeStr.c_str(), "FLOAT");
        return ge::GRAPH_FAILED;
    }

    auto prevSoftmaxSumTensor = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxSumTensor);
    ge::DataType prevSoftmaxSumDtype = prevSoftmaxSumTensor->GetDataType();
    if (prevSoftmaxSumDtype != ge::DT_FLOAT) {
        std::string prevSoftmaxSumDtypeStr = ToString(prevSoftmaxSumDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "prev_softmax_sum", prevSoftmaxSumDtypeStr.c_str(), "FLOAT");
        return ge::GRAPH_FAILED;
    }

    auto curAttnTensor = context->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, curAttnTensor);
    ge::DataType curAttnDtype = curAttnTensor->GetDataType();
    if (curAttnDtype != ge::DT_FLOAT16 && curAttnDtype != ge::DT_BF16 && curAttnDtype != ge::DT_FLOAT) {
        std::string curAttnDtypeStr = ToString(curAttnDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "cur_attn_out", curAttnDtypeStr.c_str(),
                                  "FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }

    auto curSoftmaxMaxTensor = context->GetInputDesc(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, curSoftmaxMaxTensor);
    ge::DataType curSoftmaxMaxDtype = curSoftmaxMaxTensor->GetDataType();
    if (curSoftmaxMaxDtype != ge::DT_FLOAT) {
        std::string curSoftmaxMaxDtypeStr = ToString(curSoftmaxMaxDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "cur_softmax_max", curSoftmaxMaxDtypeStr.c_str(), "FLOAT");
        return ge::GRAPH_FAILED;
    }

    auto curSoftmaxSumTensor = context->GetInputDesc(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, curSoftmaxSumTensor);
    ge::DataType curSoftmaxSumDtype = curSoftmaxSumTensor->GetDataType();
    if (curSoftmaxSumDtype != ge::DT_FLOAT) {
        std::string curSoftmaxSumDtypeStr = ToString(curSoftmaxSumDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "cur_softmax_sum", curSoftmaxSumDtypeStr.c_str(), "FLOAT");
        return ge::GRAPH_FAILED;
    }

    // 输入间的dtype约束
    if (prevAttnDtype != curAttnDtype) {
        std::string dtypeMsg = ToString(prevAttnDtype) + " and " + ToString(curAttnDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(), "prev_attn_out and cur_attn_out", dtypeMsg.c_str(),
            "The dtypes of input prev_attn_out and input cur_attn_out should be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseSBHCheckInputShape(const gert::TilingContext* context)
{
    auto prevAttnShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevAttnShapePtr);
    gert::Shape prevAttnShape = prevAttnShapePtr->GetStorageShape();
    if (prevAttnShape.GetDimNum() != CONST_THREE) {
        std::string prevAttnDimNumStr = std::to_string(prevAttnShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "prev_attn_out", prevAttnDimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }

    auto prevSoftmaxMaxShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxMaxShapePtr);
    gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();
    if (prevSoftmaxMaxShape.GetDimNum() != CONST_FOUR) {
        std::string prevSoftmaxMaxDimNumStr = std::to_string(prevSoftmaxMaxShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "prev_softmax_max", prevSoftmaxMaxDimNumStr.c_str(), "4D");
        return ge::GRAPH_FAILED;
    }
    if (prevSoftmaxMaxShape.GetDim(CONST_THREE) != SOFTMAX_TAIL) {
        std::string reasonMsg = "The last dimension of prev_softmax_max should be " + std::to_string(SOFTMAX_TAIL);
        std::string prevSoftmaxMaxShapeStr = ToString(prevSoftmaxMaxShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "prev_softmax_max",
                                              prevSoftmaxMaxShapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto prevSoftmaxSumShapePtr = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxSumShapePtr);
    gert::Shape prevSoftmaxSumShape = prevSoftmaxSumShapePtr->GetStorageShape();
    if (prevSoftmaxSumShape.GetDimNum() != CONST_FOUR) {
        std::string prevSoftmaxSumDimNumStr = std::to_string(prevSoftmaxSumShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "prev_softmax_sum", prevSoftmaxSumDimNumStr.c_str(), "4D");
        return ge::GRAPH_FAILED;
    }
    if (prevSoftmaxSumShape.GetDim(CONST_THREE) != SOFTMAX_TAIL) {
        std::string reasonMsg = "The last dimension of prev_softmax_sum should be " + std::to_string(SOFTMAX_TAIL);
        std::string prevSoftmaxSumShapeStr = ToString(prevSoftmaxSumShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "prev_softmax_sum",
                                              prevSoftmaxSumShapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto curAttnShapePtr = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, curAttnShapePtr);
    gert::Shape curAttnShape = curAttnShapePtr->GetStorageShape();
    if (curAttnShape.GetDimNum() != CONST_THREE) {
        std::string curAttnDimNumStr = std::to_string(curAttnShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "cur_attn_out", curAttnDimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }

    auto curSoftmaxMaxShapePtr = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, curSoftmaxMaxShapePtr);
    gert::Shape curSoftmaxMaxShape = curSoftmaxMaxShapePtr->GetStorageShape();
    if (curSoftmaxMaxShape.GetDimNum() != CONST_FOUR) {
        std::string curSoftmaxMaxDimNumStr = std::to_string(curSoftmaxMaxShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "cur_softmax_max", curSoftmaxMaxDimNumStr.c_str(), "4D");
        return ge::GRAPH_FAILED;
    }
    if (curSoftmaxMaxShape.GetDim(CONST_THREE) != SOFTMAX_TAIL) {
        std::string reasonMsg = "The last dimension of cur_softmax_max should be " + std::to_string(SOFTMAX_TAIL);
        std::string curSoftmaxMaxShapeStr = ToString(curSoftmaxMaxShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "cur_softmax_max", curSoftmaxMaxShapeStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto curSoftmaxSumShapePtr = context->GetInputShape(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, curSoftmaxSumShapePtr);
    gert::Shape curSoftmaxSumShape = curSoftmaxSumShapePtr->GetStorageShape();
    if (curSoftmaxSumShape.GetDimNum() != CONST_FOUR) {
        std::string curSoftmaxSumDimNumStr = std::to_string(curSoftmaxSumShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "cur_softmax_sum", curSoftmaxSumDimNumStr.c_str(), "4D");
        return ge::GRAPH_FAILED;
    }
    if (curSoftmaxSumShape.GetDim(CONST_THREE) != SOFTMAX_TAIL) {
        std::string reasonMsg = "The last dimension of cur_softmax_sum should be " + std::to_string(SOFTMAX_TAIL);
        std::string curSoftmaxSumShapeStr = ToString(curSoftmaxSumShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "cur_softmax_sum", curSoftmaxSumShapeStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseSBHCheckShape(const gert::TilingContext* context)
{
    // 输入shape校验
    OP_CHECK_IF(RingAttentionUpdateRegbaseSBHCheckInputShape(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseSBHCheckInputShape failed, tiling failed"),
                return ge::GRAPH_FAILED);

    // 输入间shape校验
    gert::Shape prevAttnShape = context->GetInputShape(0)->GetStorageShape();
    gert::Shape prevSoftmaxMaxShape = context->GetInputShape(1)->GetStorageShape();
    gert::Shape prevSoftmaxSumShape = context->GetInputShape(2)->GetStorageShape();
    gert::Shape curAttnShape = context->GetInputShape(3)->GetStorageShape();
    gert::Shape curSoftmaxMaxShape = context->GetInputShape(4)->GetStorageShape();
    gert::Shape curSoftmaxSumShape = context->GetInputShape(5)->GetStorageShape();

    if (prevAttnShape.GetDim(CONST_ZERO) != prevSoftmaxMaxShape.GetDim(CONST_TWO) ||
        prevAttnShape.GetDim(CONST_ONE) != prevSoftmaxMaxShape.GetDim(CONST_ZERO)) {
        std::string shapeMsg = ToString(prevAttnShape) + " and " + ToString(prevSoftmaxMaxShape);
        std::string reasonMsg =
            "The 0th dim of input prev_attn_out should be equal to the 2nd dim of input prev_softmax_max, "
            "and the 1st dim of input prev_attn_out should be equal to the 0th dim of input prev_softmax_max";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "prev_attn_out and prev_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (prevSoftmaxMaxShape.GetDim(CONST_ZERO) != prevSoftmaxSumShape.GetDim(CONST_ZERO) ||
        prevSoftmaxMaxShape.GetDim(CONST_ONE) != prevSoftmaxSumShape.GetDim(CONST_ONE) ||
        prevSoftmaxMaxShape.GetDim(CONST_TWO) != prevSoftmaxSumShape.GetDim(CONST_TWO)) {
        std::string shapeMsg = ToString(prevSoftmaxSumShape) + " and " + ToString(prevSoftmaxMaxShape);
        std::string reasonMsg = "The shape of input prev_softmax_sum should be the same as "
                                "the shape of input prev_softmax_max";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "prev_softmax_sum and prev_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (curAttnShape.GetDim(CONST_ZERO) != curSoftmaxMaxShape.GetDim(CONST_TWO) ||
        curAttnShape.GetDim(CONST_ONE) != curSoftmaxMaxShape.GetDim(CONST_ZERO)) {
        std::string shapeMsg = ToString(curAttnShape) + " and " + ToString(curSoftmaxMaxShape);
        std::string reasonMsg =
            "The 0th dim of input cur_attn_out should be equal to the 2nd dim of input cur_softmax_max, "
            "and the 1st dim of input cur_attn_out should be equal to the 0th dim of input cur_softmax_max";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "cur_attn_out and cur_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (curSoftmaxMaxShape.GetDim(CONST_ZERO) != curSoftmaxSumShape.GetDim(CONST_ZERO) ||
        curSoftmaxMaxShape.GetDim(CONST_ONE) != curSoftmaxSumShape.GetDim(CONST_ONE) ||
        curSoftmaxMaxShape.GetDim(CONST_TWO) != curSoftmaxSumShape.GetDim(CONST_TWO)) {
        std::string shapeMsg = ToString(curSoftmaxSumShape) + " and " + ToString(curSoftmaxMaxShape);
        std::string reasonMsg =
            "The first three dimensions of input cur_softmax_sum and cur_softmax_max should be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "cur_softmax_sum and cur_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (prevSoftmaxMaxShape.GetDim(CONST_ZERO) != curSoftmaxMaxShape.GetDim(CONST_ZERO) ||
        prevSoftmaxMaxShape.GetDim(CONST_ONE) != curSoftmaxMaxShape.GetDim(CONST_ONE) ||
        prevSoftmaxMaxShape.GetDim(CONST_TWO) != curSoftmaxMaxShape.GetDim(CONST_TWO)) {
        std::string shapeMsg = ToString(prevSoftmaxMaxShape) + " and " + ToString(curSoftmaxMaxShape);
        std::string reasonMsg =
            "The first three dimensions of input prev_softmax_max and cur_softmax_max should be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "prev_softmax_max and cur_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseTNDCheckInputShape(const gert::TilingContext* context)
{
    auto prevAttnShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevAttnShapePtr);
    gert::Shape prevAttnShape = prevAttnShapePtr->GetStorageShape();
    if (prevAttnShape.GetDimNum() != CONST_THREE) {
        std::string prevAttnDimNumStr = std::to_string(prevAttnShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "prev_attn_out", prevAttnDimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }

    auto prevSoftmaxMaxShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxMaxShapePtr);
    gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();
    if (prevSoftmaxMaxShape.GetDimNum() != CONST_THREE) {
        std::string prevSoftmaxMaxDimNumStr = std::to_string(prevSoftmaxMaxShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "prev_softmax_max", prevSoftmaxMaxDimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }
    if (prevSoftmaxMaxShape.GetDim(CONST_TWO) != SOFTMAX_TAIL) {
        std::string reasonMsg = "The last dimension of prev_softmax_max should be " + std::to_string(SOFTMAX_TAIL);
        std::string prevSoftmaxMaxShapeStr = ToString(prevSoftmaxMaxShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "prev_softmax_max",
                                              prevSoftmaxMaxShapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto prevSoftmaxSumShapePtr = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxSumShapePtr);
    gert::Shape prevSoftmaxSumShape = prevSoftmaxSumShapePtr->GetStorageShape();
    if (prevSoftmaxSumShape.GetDimNum() != CONST_THREE) {
        std::string prevSoftmaxSumDimNumStr = std::to_string(prevSoftmaxSumShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "prev_softmax_sum", prevSoftmaxSumDimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }
    if (prevSoftmaxSumShape.GetDim(CONST_TWO) != SOFTMAX_TAIL) {
        std::string reasonMsg = "The last dimension of prev_softmax_sum should be " + std::to_string(SOFTMAX_TAIL);
        std::string prevSoftmaxSumShapeStr = ToString(prevSoftmaxSumShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "prev_softmax_sum",
                                              prevSoftmaxSumShapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto curAttnShapePtr = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, curAttnShapePtr);
    gert::Shape curAttnShape = curAttnShapePtr->GetStorageShape();
    if (curAttnShape.GetDimNum() != CONST_THREE) {
        std::string curAttnDimNumStr = std::to_string(curAttnShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "cur_attn_out", curAttnDimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }

    auto curSoftmaxMaxShapePtr = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, curSoftmaxMaxShapePtr);
    gert::Shape curSoftmaxMaxShape = curSoftmaxMaxShapePtr->GetStorageShape();
    if (curSoftmaxMaxShape.GetDimNum() != CONST_THREE) {
        std::string curSoftmaxMaxDimNumStr = std::to_string(curSoftmaxMaxShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "cur_softmax_max", curSoftmaxMaxDimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }
    if (curSoftmaxMaxShape.GetDim(CONST_TWO) != SOFTMAX_TAIL) {
        std::string reasonMsg = "The last dimension of cur_softmax_max should be " + std::to_string(SOFTMAX_TAIL);
        std::string curSoftmaxMaxShapeStr = ToString(curSoftmaxMaxShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "cur_softmax_max", curSoftmaxMaxShapeStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto curSoftmaxSumShapePtr = context->GetInputShape(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, curSoftmaxSumShapePtr);
    gert::Shape curSoftmaxSumShape = curSoftmaxSumShapePtr->GetStorageShape();
    if (curSoftmaxSumShape.GetDimNum() != CONST_THREE) {
        std::string curSoftmaxSumDimNumStr = std::to_string(curSoftmaxSumShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "cur_softmax_sum", curSoftmaxSumDimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }
    if (curSoftmaxSumShape.GetDim(CONST_TWO) != SOFTMAX_TAIL) {
        std::string reasonMsg = "The last dimension of cur_softmax_sum should be " + std::to_string(SOFTMAX_TAIL);
        std::string curSoftmaxSumShapeStr = ToString(curSoftmaxSumShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "cur_softmax_sum", curSoftmaxSumShapeStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseTNDCheckShape(const gert::TilingContext* context)
{
    // 输入shape校验
    OP_CHECK_IF(RingAttentionUpdateRegbaseTNDCheckInputShape(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseTNDCheckInputShape failed, tiling failed"),
                return ge::GRAPH_FAILED);

    // 输入间shape约束
    gert::Shape prevAttnShape = context->GetInputShape(0)->GetStorageShape();
    gert::Shape prevSoftmaxMaxShape = context->GetInputShape(1)->GetStorageShape();
    gert::Shape prevSoftmaxSumShape = context->GetInputShape(2)->GetStorageShape();
    gert::Shape curAttnShape = context->GetInputShape(3)->GetStorageShape();
    gert::Shape curSoftmaxMaxShape = context->GetInputShape(4)->GetStorageShape();
    gert::Shape curSoftmaxSumShape = context->GetInputShape(5)->GetStorageShape();

    if (prevAttnShape.GetDim(CONST_ZERO) != prevSoftmaxMaxShape.GetDim(CONST_ZERO) ||
        prevAttnShape.GetDim(CONST_ONE) != prevSoftmaxMaxShape.GetDim(CONST_ONE)) {
        std::string shapeMsg = ToString(prevAttnShape) + " and " + ToString(prevSoftmaxMaxShape);
        std::string reasonMsg =
            "The first two dimensions of input prev_attn_out and prev_softmax_max should be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "prev_attn_out and prev_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (prevSoftmaxMaxShape.GetDim(CONST_ZERO) != prevSoftmaxSumShape.GetDim(CONST_ZERO) ||
        prevSoftmaxMaxShape.GetDim(CONST_ONE) != prevSoftmaxSumShape.GetDim(CONST_ONE)) {
        std::string shapeMsg = ToString(prevSoftmaxSumShape) + " and " + ToString(prevSoftmaxMaxShape);
        std::string reasonMsg =
            "The first two dimensions of input prev_softmax_sum and prev_softmax_max should be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "prev_softmax_sum and prev_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (curAttnShape.GetDim(CONST_ZERO) != curSoftmaxMaxShape.GetDim(CONST_ZERO) ||
        curAttnShape.GetDim(CONST_ONE) != curSoftmaxMaxShape.GetDim(CONST_ONE)) {
        std::string shapeMsg = ToString(curAttnShape) + " and " + ToString(curSoftmaxMaxShape);
        std::string reasonMsg = "The first two dimensions of input cur_attn_out and cur_softmax_max should be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "cur_attn_out and cur_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (curSoftmaxMaxShape.GetDim(CONST_ZERO) != curSoftmaxSumShape.GetDim(CONST_ZERO) ||
        curSoftmaxMaxShape.GetDim(CONST_ONE) != curSoftmaxSumShape.GetDim(CONST_ONE)) {
        std::string shapeMsg = ToString(curSoftmaxSumShape) + " and " + ToString(curSoftmaxMaxShape);
        std::string reasonMsg =
            "The first two dimensions of input cur_softmax_sum and cur_softmax_max should be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "cur_softmax_sum and cur_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (prevSoftmaxMaxShape.GetDim(CONST_ZERO) != curSoftmaxMaxShape.GetDim(CONST_ZERO) ||
        prevSoftmaxMaxShape.GetDim(CONST_ONE) != curSoftmaxMaxShape.GetDim(CONST_ONE)) {
        std::string shapeMsg = ToString(prevSoftmaxMaxShape) + " and " + ToString(curSoftmaxMaxShape);
        std::string reasonMsg =
            "The first two dimensions of input prev_softmax_max and cur_softmax_max should be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "prev_softmax_max and cur_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void RingAttentionUpdatePrintParamRegbaseSoftmaxTND(const gert::TilingContext* context,
                                                           RingAttentionUpdateRegbaseSoftmaxTNDTilingData& tiling)
{
    // output param
    OP_LOGD(context->GetNodeName(), "dimT = %ld", tiling.get_dimT());
    OP_LOGD(context->GetNodeName(), "dimN = %ld", tiling.get_dimN());
    OP_LOGD(context->GetNodeName(), "dimD = %ld", tiling.get_dimD());
    OP_LOGD(context->GetNodeName(), "softmaxTailSize = %ld", tiling.get_softmaxTailSize());
    OP_LOGD(context->GetNodeName(), "batchSize = %ld", tiling.get_batchSize());

    OP_LOGD(context->GetNodeName(), "usedVectorCoreNum = %ld", tiling.get_usedVectorCoreNum());
    OP_LOGD(context->GetNodeName(), "tnNumCoreEach = %ld", tiling.get_tnNumCoreEach());
    OP_LOGD(context->GetNodeName(), "tnNumCoreTail = %ld", tiling.get_tnNumCoreTail());

    OP_LOGD(context->GetNodeName(), "tnNumLoopEach = %ld", tiling.get_tnNumLoopEach());
    OP_LOGD(context->GetNodeName(), "headDimLoopEach = %ld", tiling.get_headDimLoopEach());
}

static void RingAttentionUpdatePrintParamRegbaseTND(const gert::TilingContext* context,
                                                    RingAttentionUpdateRegbaseTNDTilingData& tiling)
{
    // output param
    OP_LOGD(context->GetNodeName(), "dimT = %ld", tiling.get_dimT());
    OP_LOGD(context->GetNodeName(), "dimN = %ld", tiling.get_dimN());
    OP_LOGD(context->GetNodeName(), "dimD = %ld", tiling.get_dimD());
    OP_LOGD(context->GetNodeName(), "softmaxTailSize = %ld", tiling.get_softmaxTailSize());
    OP_LOGD(context->GetNodeName(), "batchSize = %ld", tiling.get_batchSize());

    OP_LOGD(context->GetNodeName(), "usedVectorCoreNum = %ld", tiling.get_usedVectorCoreNum());
    OP_LOGD(context->GetNodeName(), "dimTCoreEach = %ld", tiling.get_dimTCoreEach());
    OP_LOGD(context->GetNodeName(), "dimTCoreTail = %ld", tiling.get_dimTCoreTail());

    OP_LOGD(context->GetNodeName(), "seqNumLoopEach = %ld", tiling.get_seqNumLoopEach());
    OP_LOGD(context->GetNodeName(), "headNumLoopEach = %ld", tiling.get_headNumLoopEach());
    OP_LOGD(context->GetNodeName(), "headDimLoopEach = %ld", tiling.get_headDimLoopEach());
}

static void RingAttentionUpdatePrintParamRegbaseSBH(const gert::TilingContext* context,
                                                    RingAttentionUpdateRegbaseSBHTilingData& tiling)
{
    // output param
    OP_LOGD(context->GetNodeName(), "dimB = %ld", tiling.get_dimB());
    OP_LOGD(context->GetNodeName(), "dimS = %ld", tiling.get_dimS());
    OP_LOGD(context->GetNodeName(), "dimN = %ld", tiling.get_dimN());
    OP_LOGD(context->GetNodeName(), "dimD = %ld", tiling.get_dimD());
    OP_LOGD(context->GetNodeName(), "softmaxTailSize = %ld", tiling.get_softmaxTailSize());

    OP_LOGD(context->GetNodeName(), "usedVectorCoreNum = %ld", tiling.get_usedVectorCoreNum());
    OP_LOGD(context->GetNodeName(), "coreNumGroup = %ld", tiling.get_coreNumGroup());
    OP_LOGD(context->GetNodeName(), "bnNumGroup = %ld", tiling.get_bnNumGroup());
    OP_LOGD(context->GetNodeName(), "seqNumCoreEach = %ld", tiling.get_seqNumCoreEach());
    OP_LOGD(context->GetNodeName(), "seqNumCoreTail = %ld", tiling.get_seqNumCoreTail());

    OP_LOGD(context->GetNodeName(), "seqNumLoopEach = %ld", tiling.get_seqNumLoopEach());
    OP_LOGD(context->GetNodeName(), "headDimLoopEach = %ld", tiling.get_headDimLoopEach());
}

static int64_t GetInputDtypeSize(const gert::TilingContext* context)
{
    int64_t inputDtypeSize = 0;
    auto attnTensor = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attnTensor);
    auto attnDtype = attnTensor->GetDataType();
    if (attnDtype == ge::DT_FLOAT) {
        inputDtypeSize = SIZE_B32;
    } else if (attnDtype == ge::DT_FLOAT16 || attnDtype == ge::DT_BF16) {
        inputDtypeSize = SIZE_B16;
    } else { // return 0
        std::string attnDtypeStr = ToString(attnDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "prev_attn_out", attnDtypeStr.c_str(),
                                  "FLOAT, FLOAT16 or BF16");
    }
    OP_LOGD(context->GetNodeName(), "input data type size = %d", inputDtypeSize);
    return inputDtypeSize;
}

static ge::graphStatus RingAttentionUpdateRegbaseSoftmaxTNDSplitCore(
    const gert::TilingContext* context, RingAttentionUpdateRegbaseSoftmaxTNDTilingData& tiling)
{
    int64_t dimT = tiling.get_dimT();
    int64_t dimN = tiling.get_dimN();
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();

    int64_t tnNumCoreEach = CeilDiv<int64_t>(dimT * dimN, maxCoreNum);             // 每核处理的TN方向长度
    int64_t usedVectorCoreNum = CeilDiv<int64_t>(dimT * dimN, tnNumCoreEach);      // 使用的vector核数
    int64_t tnNumCoreTail = dimT * dimN - tnNumCoreEach * (usedVectorCoreNum - 1); // 尾核处理的TN方向长度

    tiling.set_tnNumCoreEach(tnNumCoreEach);
    tiling.set_tnNumCoreTail(tnNumCoreTail);
    tiling.set_usedVectorCoreNum(usedVectorCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseSoftmaxTNDSplitLoop(
    const gert::TilingContext* context, RingAttentionUpdateRegbaseSoftmaxTNDTilingData& tiling)
{
    uint64_t ubSize = 0;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t dimD = tiling.get_dimD();
    int64_t softmaxTailSize = tiling.get_softmaxTailSize();
    int64_t inputDtypeSize = GetInputDtypeSize(context);
    OP_CHECK_IF(inputDtypeSize <= 0, OP_LOGE(context->GetNodeName(), "Dtype only support fp16, fp32, bf16 currently."),
                return ge::GRAPH_FAILED);

    // 核内循环，UB空间分配
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (prev_softmax_max_input)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (cur_softmax_max_input)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (prev_softmax_sum_input)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (cur_softmax_sum_input)
    // loop * D_align * dtype * DOUBLE_BUFFER; (prev_attn_out_input)
    // loop * D_align * dtype * DOUBLE_BUFFER; (cur_attn_out_input)
    // loop * D_align * dtype * DOUBLE_BUFFER; (attn_out_output)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (softmax_max_output)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (softmax_sum_output)
    // loop * 8 * FLOAT_DTYPE; (cur_out_scale_ub)
    // loop * 8 * FLOAT_DTYPE; (pre_out_scale_ub)
    int64_t doubleBufferNum = 2;
    int64_t softmaxQueueNum = 4 + 2;
    int64_t softmaxBufferTempNum = 2;
    int64_t attnQueueNum = 2 + 1;
    int64_t softmaxEleSize = softmaxQueueNum * doubleBufferNum * SIZE_B32 + softmaxBufferTempNum * SIZE_B32;
    int64_t attnEleSize = attnQueueNum * doubleBufferNum * inputDtypeSize;

    OP_CHECK_IF(SafeDivisionCheck(softmaxTailSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Division by zero(softmaxTailSize) is not supported"),
                return ge::GRAPH_FAILED);
    int64_t tnNumLoopMin = REPEAT_NUM_B32 / softmaxTailSize; // TN方向至少搬入长度，避免softmax单次搬运数据量过少(256B)
    int64_t softmaxSizeMin = tnNumLoopMin * softmaxTailSize * softmaxEleSize; // softmax相关部分最少UB空间大小
    int64_t ubSizeRemain = ubSize - softmaxSizeMin;

    int64_t headDimLoopAlign = REPEAT_NUM_B32;
    int64_t headDimLoopMax = ubSizeRemain / (tnNumLoopMin * attnEleSize);
    headDimLoopMax = headDimLoopMax / headDimLoopAlign * headDimLoopAlign; // 对齐, UB能放下的最大支持headdim长度
    int64_t tnNumLoopEach;                                                 // 单次UB循环TN长度
    int64_t headDimLoopEach;                                               // 单次UB循环headDim长度
    if (dimD < headDimLoopMax) {
        headDimLoopEach = (dimD + headDimLoopAlign - 1) / headDimLoopAlign * headDimLoopAlign;
        tnNumLoopEach = ubSize / (softmaxTailSize * softmaxEleSize + headDimLoopEach * attnEleSize);
    } else {
        headDimLoopEach = headDimLoopMax;
        tnNumLoopEach = tnNumLoopMin;
    }

    tiling.set_tnNumLoopEach(tnNumLoopEach);
    tiling.set_headDimLoopEach(headDimLoopEach);
    return ge::GRAPH_SUCCESS;
}

// SoftmaxTND分支, TN合轴分核
static ge::graphStatus Tiling4RingAttentionUpdateRegbaseSoftmaxTND(
    const gert::TilingContext* context, RingAttentionUpdateRegbaseSoftmaxTNDTilingData& tiling)
{
    // infer shape
    auto prevAttnOutShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevAttnOutShapePtr);
    gert::Shape prevAttnOutShape = prevAttnOutShapePtr->GetStorageShape();

    auto prevSoftmaxMaxShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxMaxShapePtr);
    gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();

    auto actualSeqQlenShapePtr = context->GetInputShape(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualSeqQlenShapePtr);
    gert::Shape actualSeqQlenShape = actualSeqQlenShapePtr->GetStorageShape();

    int64_t batchSize = actualSeqQlenShape.GetDim(0) - 1;
    int64_t dimT = prevAttnOutShape.GetDim(0);
    int64_t dimN = prevAttnOutShape.GetDim(1);
    int64_t dimD = prevAttnOutShape.GetDim(2);
    int64_t softmaxTailSize = prevSoftmaxMaxShape.GetDim(2);
    if (batchSize <= 0) {
        std::string actualSeqQlenShapeStr = ToString(actualSeqQlenShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "actual_seq_qlen", actualSeqQlenShapeStr.c_str(),
                                              "The batch of input actual_seq_qlen should be greater than 1, "
                                              "where batch refers to the 0th dim");
        return ge::GRAPH_FAILED;
    }

    tiling.set_dimT(dimT);
    tiling.set_dimN(dimN);
    tiling.set_dimD(dimD);
    tiling.set_softmaxTailSize(softmaxTailSize);
    tiling.set_batchSize(batchSize);

    // 核间分核
    OP_CHECK_IF(RingAttentionUpdateRegbaseSoftmaxTNDSplitCore(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseSoftmaxTNDSplitCore failed."),
                return ge::GRAPH_FAILED);

    // 核内循环
    OP_CHECK_IF(RingAttentionUpdateRegbaseSoftmaxTNDSplitLoop(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseSoftmaxTNDSplitLoop failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseTNDSplitCore(const gert::TilingContext* context,
                                                              RingAttentionUpdateRegbaseTNDTilingData& tiling)
{
    int64_t dimT = tiling.get_dimT();
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();

    int64_t dimTCoreEach = CeilDiv<int64_t>(dimT, maxCoreNum);            // 每核处理的T方向长度
    int64_t usedVectorCoreNum = CeilDiv<int64_t>(dimT, dimTCoreEach);     // 使用的vector核数
    int64_t dimTCoreTail = dimT - dimTCoreEach * (usedVectorCoreNum - 1); // 尾核处理的T方向长度

    tiling.set_dimTCoreEach(dimTCoreEach);
    tiling.set_dimTCoreTail(dimTCoreTail);
    tiling.set_usedVectorCoreNum(usedVectorCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseTNDSplitLoop(const gert::TilingContext* context,
                                                              RingAttentionUpdateRegbaseTNDTilingData& tiling)
{
    uint64_t ubSizeTotal = 0;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTotal);
    int64_t dimN = tiling.get_dimN();
    int64_t dimD = tiling.get_dimD();
    int64_t softmaxTailSize = tiling.get_softmaxTailSize();
    int64_t inputDtypeSize = GetInputDtypeSize(context);
    OP_CHECK_IF(inputDtypeSize <= 0, OP_LOGE(context->GetNodeName(), "Dtype only support fp16, fp32, bf16 currently."),
                return ge::GRAPH_FAILED);

    // 核内循环，UB空间分配
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (prev_softmax_max_input)
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (cur_softmax_max_input)
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (prev_softmax_sum_input)
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (cur_softmax_sum_input)
    // seqLoop * headNumLoop * D_align * dtype * DOUBLE_BUFFER; (prev_attn_out_input)
    // seqLoop * headNumLoop * D_align * dtype * DOUBLE_BUFFER; (cur_attn_out_input)
    // seqLoop * headNumLoop * D_align * dtype * DOUBLE_BUFFER; (attn_out_output)
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (softmax_max_output)
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (softmax_sum_output)
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE; (cur_out_scale_ub)
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE; (pre_out_scale_ub)
    // 如果ND很小，UB可以载入多个Seq，UB内需要重排，额外申请下面2片空间
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE; (cur_out_scale_ub重排)
    // seqLoop * headNumLoop * 8 * FLOAT_DTYPE; (pre_out_scale_ub重排)
    int64_t seqNumLoopEach = 1; // 单次UB循环Seq长度
    int64_t headNumLoopEach;    // 单次UB循环headNum长度
    int64_t headDimLoopEach;    // 单次UB循环headDim长度

    int64_t seqNumLoopMin = 1; // Seq方向单次搬入1长度
    int64_t doubleBufferNum = 2;
    int64_t softmaxQueueNum = 4 + 2;
    int64_t softmaxBufferTempNum = 2 + 2; // 包含重排的UB
    int64_t attnQueueNum = 2 + 1;
    int64_t softmaxEleSize = softmaxQueueNum * doubleBufferNum * SIZE_B32 + softmaxBufferTempNum * SIZE_B32; // 64
    int64_t attnEleSize = attnQueueNum * doubleBufferNum * inputDtypeSize;                                   // 12 or 24
    int64_t headDimAlign = (dimD + REPEAT_NUM_B32 - 1) / REPEAT_NUM_B32 * REPEAT_NUM_B32;
    int64_t ubSize = (int64_t)ubSizeTotal;

    if (seqNumLoopMin * dimN * softmaxTailSize * softmaxEleSize + seqNumLoopMin * dimN * headDimAlign * attnEleSize <
        ubSize) { // UB空间可以全载ND
        OP_LOGD(context->GetNodeName(), "dimN & dimD is very small, UB can load multi Seq");
        seqNumLoopEach = ubSize / (dimN * softmaxTailSize * softmaxEleSize +
                                   dimN * headDimAlign * attnEleSize); // Seq方向单次搬入长度
        headNumLoopEach = dimN;
        headDimLoopEach = headDimAlign;
    } else {
        OP_LOGD(context->GetNodeName(), "dimN & dimD is big, UB can only load 1 Seq");
        softmaxBufferTempNum = CONST_TWO; // 无需重排
        softmaxEleSize = softmaxQueueNum * doubleBufferNum * SIZE_B32 + softmaxBufferTempNum * SIZE_B32;
        seqNumLoopEach = 1; // Seq方向单次搬入长度为1
        int64_t headNumLoopMin =
            REPEAT_NUM_B32 / softmaxTailSize; // headNum方向至少搬入长度，避免softmax单次搬运数据量过少(256B)
        int64_t softmaxSizeMin = headNumLoopMin * softmaxTailSize * softmaxEleSize; // softmax相关部分最少UB空间大小
        int64_t ubSizeRemain = ubSize - softmaxSizeMin;

        int64_t headDimLoopMax = ubSizeRemain / (headNumLoopMin * attnEleSize);
        headDimLoopMax = headDimLoopMax / REPEAT_NUM_B32 * REPEAT_NUM_B32; // 对齐
        if (dimD < headDimLoopMax) {
            headDimLoopEach = (dimD + REPEAT_NUM_B32 - 1) / REPEAT_NUM_B32 * REPEAT_NUM_B32;
            headNumLoopEach = ubSize / (softmaxTailSize * softmaxEleSize + headDimLoopEach * attnEleSize);
        } else {
            headDimLoopEach = headDimLoopMax;
            headNumLoopEach = headNumLoopMin;
        }
    }

    tiling.set_seqNumLoopEach(seqNumLoopEach);   // 单次UB循环Seq方向长度
    tiling.set_headNumLoopEach(headNumLoopEach); // 单次UB循环headNum长度
    tiling.set_headDimLoopEach(headDimLoopEach); // 单次UB循环headDim长度
    return ge::GRAPH_SUCCESS;
}

// TND分支, T分核. ND小，核内单次循环多个Seq计算；ND大，核内单次循环1个Seq计算
static ge::graphStatus Tiling4RingAttentionUpdateRegbaseTND(const gert::TilingContext* context,
                                                            RingAttentionUpdateRegbaseTNDTilingData& tiling)
{
    // infer shape
    auto actualSeqQlenShapePtr = context->GetInputShape(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualSeqQlenShapePtr);
    gert::Shape actualSeqQlenShape = actualSeqQlenShapePtr->GetStorageShape();

    auto prevAttnOutShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevAttnOutShapePtr);
    gert::Shape prevAttnOutShape = prevAttnOutShapePtr->GetStorageShape();

    auto prevSoftmaxMaxShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxMaxShapePtr);
    gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();

    int64_t dimT = prevAttnOutShape.GetDim(0);
    int64_t dimN = prevAttnOutShape.GetDim(1);
    int64_t dimD = prevAttnOutShape.GetDim(2);
    int64_t batchSize = actualSeqQlenShape.GetDim(0) - 1;
    int64_t softmaxTailSize = prevSoftmaxMaxShape.GetDim(2);
    if (batchSize <= 0) {
        std::string actualSeqQlenShapeStr = ToString(actualSeqQlenShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "actual_seq_qlen", actualSeqQlenShapeStr.c_str(),
                                              "The batch of input actual_seq_qlen should be greater than 1, "
                                              "where batch refers to the 0th dim");
        return ge::GRAPH_FAILED;
    }

    tiling.set_dimD(dimD);
    tiling.set_dimN(dimN);
    tiling.set_dimT(dimT);
    tiling.set_softmaxTailSize(softmaxTailSize);
    tiling.set_batchSize(batchSize);

    // 核间分核
    OP_CHECK_IF(RingAttentionUpdateRegbaseTNDSplitCore(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseTNDSplitCore failed."),
                return ge::GRAPH_FAILED);

    // 核内循环
    OP_CHECK_IF(RingAttentionUpdateRegbaseTNDSplitLoop(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseTNDSplitLoop failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static int64_t RingAttentionUpdateRegbaseGcd(int64_t inputNum0, int64_t inputNum1)
{ // 求最大公约数
    if (inputNum1 == 0) {
        return inputNum0;
    } else {
        return RingAttentionUpdateRegbaseGcd(inputNum1, inputNum0 % inputNum1);
    }
}

static ge::graphStatus RingAttentionUpdateRegbaseSBHSplitCore(const gert::TilingContext* context,
                                                              RingAttentionUpdateRegbaseSBHTilingData& tiling)
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();

    int64_t dimS = tiling.get_dimS();
    int64_t dimB = tiling.get_dimB();
    int64_t dimN = tiling.get_dimN();
    int64_t bnNum = dimB * dimN;

    int64_t groupNumTmpMax = 0;
    int64_t maxCoreNumTmp = maxCoreNum;
    for (int64_t core = maxCoreNum; core >= 1; core--) { // 适当减少核数，找到分组组数最大的情况
        int64_t groupNumTmp = RingAttentionUpdateRegbaseGcd(bnNum, core); // 分组组数，bnNum和core的最大公约数
        if (groupNumTmp > groupNumTmpMax) {
            groupNumTmpMax = groupNumTmp;
            maxCoreNumTmp = core;
        }
    }
    int64_t groupNum = groupNumTmpMax; // 分组组数
    maxCoreNum = maxCoreNumTmp;        // 核数
    OP_LOGD(context->GetNodeName(), "groupNum = %ld", groupNum);

    OP_CHECK_IF(SafeDivisionCheck(groupNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Division by zero(groupNum) is not supported"),
                return ge::GRAPH_FAILED);
    int64_t coreNumGroup = maxCoreNum / groupNum; // 每组coreNumGroup个核
    int64_t bnNumGroup = bnNum / groupNum;        // 每组计算bnNumGroup个BN

    int64_t seqNumCoreEach = CeilDiv<int64_t>(dimS, coreNumGroup);       // 每核处理的Seq长度
    coreNumGroup = CeilDiv<int64_t>(dimS, seqNumCoreEach);               // 更新每组核数
    int64_t seqNumCoreTail = dimS - (coreNumGroup - 1) * seqNumCoreEach; // 尾核处理的Seq长度
    int64_t usedVectorCoreNum = coreNumGroup * groupNum;                 // 实际使用的核数

    tiling.set_usedVectorCoreNum(usedVectorCoreNum);
    tiling.set_coreNumGroup(coreNumGroup);
    tiling.set_bnNumGroup(bnNumGroup);
    tiling.set_seqNumCoreEach(seqNumCoreEach);
    tiling.set_seqNumCoreTail(seqNumCoreTail);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseSBHSplitLoop(const gert::TilingContext* context,
                                                              RingAttentionUpdateRegbaseSBHTilingData& tiling)
{
    uint64_t maxUbSize;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);

    int64_t dimD = tiling.get_dimD();
    int64_t softmaxTailSize = tiling.get_softmaxTailSize();
    int64_t inputDtypeSize = GetInputDtypeSize(context);
    OP_CHECK_IF(inputDtypeSize <= 0, OP_LOGE(context->GetNodeName(), "Dtype only support fp16, fp32, bf16 currently."),
                return ge::GRAPH_FAILED);

    // 核内循环，UB空间分配
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (prev_softmax_max_input)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (cur_softmax_max_input)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (prev_softmax_sum_input)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (cur_softmax_sum_input)
    // loop * D_align * dtype * DOUBLE_BUFFER; (prev_attn_out_input)
    // loop * D_align * dtype * DOUBLE_BUFFER; (cur_attn_out_input)
    // loop * D_align * dtype * DOUBLE_BUFFER; (attn_out_output)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (softmax_max_output)
    // loop * 8 * FLOAT_DTYPE * DOUBLE_BUFFER; (softmax_sum_output)
    // loop * 8 * FLOAT_DTYPE; (cur_out_scale_ub)
    // loop * 8 * FLOAT_DTYPE; (pre_out_scale_ub)
    int64_t doubleBufferNum = 2;
    int64_t softmaxQueueNum = 4 + 2;
    int64_t softmaxBufferTempNum = 2;
    int64_t attnQueueNum = 3;
    int64_t softmaxEleSize = softmaxQueueNum * doubleBufferNum * SIZE_B32 + softmaxBufferTempNum * SIZE_B32;
    int64_t attnEleSize = attnQueueNum * doubleBufferNum * inputDtypeSize;

    int64_t seqNumLoopMin =
        REPEAT_NUM_B32 / softmaxTailSize; // Seq方向至少搬入seqNumLoopMin长度，避免softmax单次搬运数据量过少(至少256B)
    int64_t softmaxSizeMin = seqNumLoopMin * softmaxTailSize * softmaxEleSize; // softmax相关部分最少UB空间大小
    int64_t ubSizeRemain = maxUbSize - softmaxSizeMin;

    OP_CHECK_IF(SafeDivisionCheck(seqNumLoopMin * attnEleSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Division by zero(seqNumLoopMin or attnEleSize) is not supported"),
                return ge::GRAPH_FAILED);
    int64_t headDimLoopMax = ubSizeRemain / (seqNumLoopMin * attnEleSize);
    headDimLoopMax = headDimLoopMax / REPEAT_NUM_B32 * REPEAT_NUM_B32; // 对齐
    int64_t seqNumLoopEach;                                            // 单次UB循环Seq长度
    int64_t headDimLoopEach;                                           // 单次UB循环headDim长度
    if (dimD < headDimLoopMax) {
        headDimLoopEach = (dimD + REPEAT_NUM_B32 - 1) / REPEAT_NUM_B32 * REPEAT_NUM_B32;
        seqNumLoopEach = maxUbSize / (softmaxTailSize * softmaxEleSize + headDimLoopEach * attnEleSize);
    } else {
        headDimLoopEach = headDimLoopMax;
        seqNumLoopEach = seqNumLoopMin;
    }

    tiling.set_seqNumLoopEach(seqNumLoopEach);
    tiling.set_headDimLoopEach(headDimLoopEach);
    return ge::GRAPH_SUCCESS;
}

// SBH分支，BN和core_num确定分组，每组内Seq分核
static ge::graphStatus Tiling4RingAttentionUpdateRegbaseSBH(const gert::TilingContext* context,
                                                            RingAttentionUpdateRegbaseSBHTilingData& tiling)
{
    // infer shape
    auto prevAttnOutShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevAttnOutShapePtr);
    gert::Shape prevAttnOutShape = prevAttnOutShapePtr->GetStorageShape();

    auto prevSoftmaxMaxShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, prevSoftmaxMaxShapePtr);
    gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();

    int64_t dimB = prevSoftmaxMaxShape.GetDim(0);
    int64_t dimN = prevSoftmaxMaxShape.GetDim(1);
    int64_t dimS = prevSoftmaxMaxShape.GetDim(2);
    int64_t dimH = prevAttnOutShape.GetDim(2);
    if (SafeDivisionCheck(dimN) != ge::GRAPH_SUCCESS) {
        std::string prevSoftmaxMaxShapeStr = ToString(prevSoftmaxMaxShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "prev_softmax_max", prevSoftmaxMaxShapeStr.c_str(),
            "The N axis of input prev_softmax_max can not be 0, where N refers to the 1st dim of prev_softmax_max");
        return ge::GRAPH_FAILED;
    }
    if (dimH % dimN != 0) {
        std::string shapeMsg = ToString(prevAttnOutShape) + " and " + ToString(prevSoftmaxMaxShape);
        std::string reasonMsg =
            "The H axis of input prev_attn_out must be divisible by the N axis of input prev_softmax_max, "
            "where H refers to 2nd dim of prev_attn_out and N refers to the 1st dim of prev_softmax_max";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "prev_attn_out and prev_softmax_max",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t dimD = dimH / dimN;
    int64_t softmaxTailSize = prevSoftmaxMaxShape.GetDim(3);

    tiling.set_dimB(dimB);
    tiling.set_dimS(dimS);
    tiling.set_dimN(dimN);
    tiling.set_dimD(dimD);
    tiling.set_softmaxTailSize(softmaxTailSize);

    // 核间分核
    OP_LOGD(context->GetNodeName(), "RingAttentionUpdateRegbaseSBHTiling RingAttentionUpdateRegbaseSBHSplitCore");
    OP_CHECK_IF(RingAttentionUpdateRegbaseSBHSplitCore(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseSBHSplitCore failed."),
                return ge::GRAPH_FAILED);

    // 核内循环
    OP_CHECK_IF(RingAttentionUpdateRegbaseSBHSplitLoop(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseSBHSplitLoop failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t GetTndSoftmaxLayout(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    int64_t tndSoftmaxLayout = 0;
    const char* inputSoftmaxLayout = attrs->GetAttrPointer<char>(1);
    if (inputSoftmaxLayout == nullptr) {
        tndSoftmaxLayout = 0;
    } else {
        std::string inputSoftmaxLayoutStr = inputSoftmaxLayout;
        if (inputSoftmaxLayoutStr == "TND") {
            tndSoftmaxLayout = 1;
        } else if (inputSoftmaxLayoutStr == "" || inputSoftmaxLayoutStr == "SBH") {
            tndSoftmaxLayout = 0;
        } else {
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "input_softmax_layout", inputSoftmaxLayoutStr.c_str(),
                                      "empty string, TND or SBH");
            tndSoftmaxLayout = -1;
        }
    }
    return tndSoftmaxLayout;
}

static ge::graphStatus RingAttentionUpdateRegbaseSoftmaxTNDTiling(gert::TilingContext* context)
{
    // init tiling data
    RingAttentionUpdateRegbaseSoftmaxTNDTilingData tiling;
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    OP_CHECK_IF(memset_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(), 0,
                         context->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(context->GetNodeName(), "Fail to clear softmaxTND tiling data"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(Tiling4RingAttentionUpdateRegbaseSoftmaxTND(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Tiling4RingAttentionUpdateRegbaseSoftmaxTND failed, tiling failed"),
                return ge::GRAPH_FAILED);
    // print tiling param
    RingAttentionUpdatePrintParamRegbaseSoftmaxTND(context, tiling);
    // init tiling setting
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(tiling.get_usedVectorCoreNum());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseTNDTiling(gert::TilingContext* context)
{
    // init tiling data
    RingAttentionUpdateRegbaseTNDTilingData tiling;
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    OP_CHECK_IF(memset_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(), 0,
                         context->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(context->GetNodeName(), "Fail to clear TND tiling data"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(Tiling4RingAttentionUpdateRegbaseTND(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Tiling4RingAttentionUpdateRegbaseTND failed, tiling failed"),
                return ge::GRAPH_FAILED);
    // print tiling param
    RingAttentionUpdatePrintParamRegbaseTND(context, tiling);
    // init tiling setting
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(tiling.get_usedVectorCoreNum());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RingAttentionUpdateRegbaseSBHTiling(gert::TilingContext* context)
{
    // init tiling data
    RingAttentionUpdateRegbaseSBHTilingData tiling;
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    OP_CHECK_IF(memset_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(), 0,
                         context->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(context->GetNodeName(), "Fail to clear SBH tiling data"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(Tiling4RingAttentionUpdateRegbaseSBH(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Tiling4RingAttentionUpdateRegbaseSBH failed, tiling failed"),
                return ge::GRAPH_FAILED);
    // print tiling param
    RingAttentionUpdatePrintParamRegbaseSBH(context, tiling);
    // init tiling setting
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(tiling.get_usedVectorCoreNum());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RingAttentionUpdateRegbase(gert::TilingContext* context)
{
    // dtype check
    OP_LOGD(context->GetNodeName(), "RingAttentionUpdateRegbaseCheckDtype start");
    OP_CHECK_IF(RingAttentionUpdateRegbaseCheckDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseCheckDtype failed."),
                return ge::GRAPH_FAILED);
    // get attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* inputLayout = attrs->GetAttrPointer<char>(0);
    OP_CHECK_IF(inputLayout == nullptr,
                OP_LOGE(context->GetNodeName(), "Get required attr input_layout failed, tiling failed"),
                return ge::GRAPH_FAILED);
    std::string inputLayoutStr = inputLayout;
    int64_t tndSoftmaxLayout = GetTndSoftmaxLayout(context);
    OP_CHECK_IF(tndSoftmaxLayout == -1,
                OP_LOGE(context->GetNodeName(), "Get required attr input_softmax_layout failed, tiling failed"),
                return ge::GRAPH_FAILED);

    uint32_t tilingKey = REGBASE_KEY;
    if (inputLayoutStr == "TND") {
        OP_LOGD(context->GetNodeName(), "Attr input_layout is TND.");
        // shape check
        OP_CHECK_IF(RingAttentionUpdateRegbaseTNDCheckShape(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseTNDCheckShape failed."),
                    return ge::GRAPH_FAILED);
        if (tndSoftmaxLayout == 1) {      // SOFTMAX_TND分支tiling
            tilingKey += SOFTMAX_TND_KEY; // +10
            OP_CHECK_IF(RingAttentionUpdateRegbaseSoftmaxTNDTiling(context) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseSoftmaxTNDTiling failed."),
                        return ge::GRAPH_FAILED);
        } else {                  // TND分支tiling
            tilingKey += TND_KEY; // +20
            OP_CHECK_IF(RingAttentionUpdateRegbaseTNDTiling(context) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseTNDTiling failed."),
                        return ge::GRAPH_FAILED);
        }
    } else { // SBH分支tiling
        OP_LOGD(context->GetNodeName(), "Attr input_layout is SBH.");
        // shape check
        OP_CHECK_IF(RingAttentionUpdateRegbaseSBHCheckShape(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseSBHCheckShape failed."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RingAttentionUpdateRegbaseSBHTiling(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "RingAttentionUpdateRegbaseSBHTiling failed."),
                    return ge::GRAPH_FAILED);
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "RingAttentionUpdateRegbaseTiling tiling end");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
