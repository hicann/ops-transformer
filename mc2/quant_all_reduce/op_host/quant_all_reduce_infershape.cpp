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
 * \file quant_all_reduce_infershape.cpp
 * \brief 图模式走infershape
 */
#include <platform/platform_info.h>
#include <register/op_impl_registry.h>
#include "op_host/mc2_common_infershape.h"
#include "mc2_log.h"
#include "common/utils/op_mc2.h"
#include "util/math_util.h"

namespace ops {

using namespace ge;

static const char *INNER_DEBUG = "MC2: QuantAllReduce InferShape Debug";
// 原型IR中input的index
constexpr size_t X_INDEX = 1;
constexpr size_t SCALES_INDEX = 2;
// 原型IR中output的index
constexpr size_t OUTPUT_INDEX = 0;
// 原型IR中attr的index
constexpr size_t OUTPUT_DTYPE_INDEX = 2;
constexpr size_t WORLD_SIZE_INDEX = 3;
// x的维度
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
// 轴信息
constexpr size_t AXIS_TWO = 2;
// rankSize有效值
const std::vector<int> SUPPORT_RANK_SIZE = {2, 4, 8};

struct QuantAllReduceShapeInfo {
    int64_t b;
    int64_t s;
    int64_t bs;
    int64_t hiddenSize;
    uint64_t xDim;
    int64_t rankNum;
};

/**
 * @brief 获取shape信息
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param shapeInfo: 临时变量存放shape信息
 */
static ge::graphStatus GetShapeInfo(const gert::InferShapeContext *context, QuantAllReduceShapeInfo &shapeInfo)
{
    const auto x_shape = context->GetInputShape(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    const auto scales_shape = context->GetInputShape(SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, scales_shape);

    const size_t x_dim = x_shape->GetDimNum();
    shapeInfo.xDim = x_dim;
    if (x_dim == DIM_TWO) {
        shapeInfo.bs = x_shape->GetDim(0);
        shapeInfo.hiddenSize = x_shape->GetDim(1);
    } else if (x_dim == DIM_THREE) {
        shapeInfo.b = x_shape->GetDim(0);
        shapeInfo.s = x_shape->GetDim(1);
        shapeInfo.bs = x_shape->GetDim(0) * x_shape->GetDim(1);
        shapeInfo.hiddenSize = x_shape->GetDim(AXIS_TWO);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(x_dim) + "D").c_str(), "2D or 3D");
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取卡数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param shapeInfo: 临时变量存放shape信息
 */
static ge::graphStatus GetRankSize(gert::InferShapeContext *context, QuantAllReduceShapeInfo &shapeInfo)
{
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 通过attr获取卡数
    const int *rankSize = attrs->GetAttrPointer<int>(WORLD_SIZE_INDEX);
    OP_LOGE_IF(rankSize == nullptr, ge::GRAPH_FAILED, context->GetNodeName(),
               "Get rank_size failed in quant_all_reduce");
    OP_TILING_CHECK(std::find(SUPPORT_RANK_SIZE.begin(), SUPPORT_RANK_SIZE.end(), *rankSize) >= SUPPORT_RANK_SIZE.end(),
                    OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "rankSize", std::to_string(*rankSize).c_str(),
                                              VectorToString(SUPPORT_RANK_SIZE).c_str()),
                    return ge::GRAPH_FAILED);

    shapeInfo.rankNum = *rankSize;
    OP_LOGD(INNER_DEBUG, "rankSize after infershape func is: %ld", shapeInfo.rankNum);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导shape，动/静态shape图逻辑一样
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 */
static ge::graphStatus InferShapeQuantAllReduce(gert::InferShapeContext *context)
{
    OPS_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(INNER_DEBUG, "Context"), return ge::GRAPH_FAILED);
    QuantAllReduceShapeInfo shapeInfo;
    // get shape_info
    OPS_CHECK(GetShapeInfo(context, shapeInfo) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Failed to get shape info in quant_all_reduce."),
              return ge::GRAPH_FAILED);

    // get rank_size
    OPS_CHECK(GetRankSize(context, shapeInfo) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Failed to get rank size in quant_all_reduce."), return ge::GRAPH_FAILED);

    auto output_shape = context->GetOutputShape(OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // infershape要的就是这些信息
    output_shape->SetDimNum(shapeInfo.xDim);
    if (shapeInfo.xDim == DIM_TWO) {
        output_shape->SetDim(0, shapeInfo.bs);
        output_shape->SetDim(1, shapeInfo.hiddenSize);
    } else {
        output_shape->SetDim(0, shapeInfo.b);
        output_shape->SetDim(1, shapeInfo.s);
        output_shape->SetDim(AXIS_TWO, shapeInfo.hiddenSize);
    }

    OP_LOGD(INNER_DEBUG, "output after infershape func, shape: [%zu], bs: [%ld], h: [%ld]", shapeInfo.xDim,
            shapeInfo.bs, shapeInfo.hiddenSize);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导dtype，动/静态shape图逻辑一样
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 */
static ge::graphStatus InferDataTypeQuantAllReduce(gert::InferDataTypeContext *context)
{
    OPS_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(INNER_DEBUG, "Context"), return ge::GRAPH_FAILED);

    OP_LOGD(INNER_DEBUG, "start to infer datatype for quant_all_reduce");
    ge::DataType get_dtype = context->GetOutputDataType(OUTPUT_INDEX);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *output_dtype_ptr = attrs->GetInt(OUTPUT_DTYPE_INDEX);
    const uint64_t output_dtype = (output_dtype_ptr != nullptr ? *output_dtype_ptr : ge::DataType::DT_BF16);
    if (output_dtype != ge::DataType::DT_BF16) {
        get_dtype = static_cast<ge::DataType>(output_dtype);
    }
    return context->SetOutputDataType(0, get_dtype);
}

IMPL_OP_INFERSHAPE(QuantAllReduce).InferShape(InferShapeQuantAllReduce).InferDataType(InferDataTypeQuantAllReduce);

} // namespace ops
