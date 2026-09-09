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
 * \file quant_reduce_scatter_infershape.cpp
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
using Ops::Base::CeilDiv;

static const char *INNER_DEBUG = "MC2: QuantReduceScatter InferShape Debug";
// 原型IR中attr的index
constexpr size_t OUTPUT_DTYPE_INDEX = 2;
constexpr size_t WORLD_SIZE_INDEX = 3;
// 原型IR中output的index
constexpr size_t OUTPUT_INDEX = 0;
// 原型IR中input的index
constexpr size_t X_INDEX = 1;
constexpr size_t SCALES_INDEX = 2;
// rankSize有效值
const std::vector<int> SUPPORT_RANK_SIZE = {2, 4, 8};
// 轴信息
constexpr size_t AXIS_TWO = 2;
constexpr int64_t DYNAMIC_SHAPE_VALUE = -1;
constexpr int64_t OUTPUT_INFER_SHAPE = 2;
// x的维度
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;

struct QuantReduceScatterShapeInfo {
    int64_t rank_num;
    uint64_t x_dim;
    int64_t b;
    int64_t s;
    int64_t bs;
    int64_t hidden_size;
};

/**
 * @brief 获取shape信息
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param shapeInfo: 临时变量存放shape信息
 */
static ge::graphStatus GetShapeInfo(const gert::InferShapeContext *context, QuantReduceScatterShapeInfo &shapeInfo)
{
    const auto scales_shape = context->GetInputShape(SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, scales_shape);

    const auto x_shape = context->GetInputShape(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    const size_t x_dim = x_shape->GetDimNum();
    shapeInfo.x_dim = x_dim;
    if (x_dim == DIM_TWO) {
        shapeInfo.bs = x_shape->GetDim(0);
        shapeInfo.hidden_size = x_shape->GetDim(1);
    } else if (x_dim == DIM_THREE) {
        shapeInfo.b = x_shape->GetDim(0);
        shapeInfo.s = x_shape->GetDim(1);
        shapeInfo.bs = x_shape->GetDim(0) * x_shape->GetDim(1);
        shapeInfo.hidden_size = x_shape->GetDim(AXIS_TWO);
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
static ge::graphStatus GetRankSize(gert::InferShapeContext *context, QuantReduceScatterShapeInfo &shapeInfo)
{
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 通过attr获取卡数
    const int *rankSize = attrs->GetAttrPointer<int>(WORLD_SIZE_INDEX);
    OP_LOGE_IF(rankSize == nullptr, ge::GRAPH_FAILED, context->GetNodeName(),
               "Get rank_size failed in quant_reduce_scatter");
    OP_TILING_CHECK(std::find(SUPPORT_RANK_SIZE.begin(), SUPPORT_RANK_SIZE.end(), *rankSize) >= SUPPORT_RANK_SIZE.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "rankSize", std::to_string(*rankSize).c_str(),
                                              VectorToString(SUPPORT_RANK_SIZE).c_str()),
                    return ge::GRAPH_FAILED);

    shapeInfo.rank_num = *rankSize;
    OP_LOGD(INNER_DEBUG, "rankSize after infershape func is: %ld", shapeInfo.rank_num);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导shape
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 */
static ge::graphStatus InferShapeQuantReduceScatter(gert::InferShapeContext *context)
{
    OPS_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(INNER_DEBUG, "Context"), return ge::GRAPH_FAILED);
    QuantReduceScatterShapeInfo shapeInfo;
    // get shape_info
    OPS_CHECK(GetShapeInfo(context, shapeInfo) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Failed to get shape info in quant_reduce_scatter."),
              return ge::GRAPH_FAILED);

    // get rank_size
    OPS_CHECK(GetRankSize(context, shapeInfo) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Failed to get rank size in quant_reduce_scatter."),
              return ge::GRAPH_FAILED);

    auto output_shape = context->GetOutputShape(OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // output始终是2维
    output_shape->SetDimNum(OUTPUT_INFER_SHAPE);
    if (shapeInfo.x_dim == DIM_THREE) {
        if (shapeInfo.b == DYNAMIC_SHAPE_VALUE) {
            // dynamic graph
            output_shape->SetDim(0, DYNAMIC_SHAPE_VALUE);
            output_shape->SetDim(1, shapeInfo.hidden_size);
        } else {
            output_shape->SetDim(0, CeilDiv(shapeInfo.b * shapeInfo.s, shapeInfo.rank_num));
            output_shape->SetDim(1, shapeInfo.hidden_size);
        }
    } else if (shapeInfo.x_dim == DIM_TWO) {
        if (shapeInfo.bs == DYNAMIC_SHAPE_VALUE) {
            // dynamic graph
            output_shape->SetDim(0, DYNAMIC_SHAPE_VALUE);
            output_shape->SetDim(1, shapeInfo.hidden_size);
        } else {
            output_shape->SetDim(0, CeilDiv(shapeInfo.bs, shapeInfo.rank_num));
            output_shape->SetDim(1, shapeInfo.hidden_size);
        }
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(shapeInfo.x_dim) + "D").c_str(),
                                     "2D or 3D");
        ge::GRAPH_FAILED;
    }

    OP_LOGD(INNER_DEBUG, "output after infershape func, shape: [%zu], bs: [%ld], h: [%ld]", shapeInfo.x_dim,
            shapeInfo.bs, shapeInfo.hidden_size);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导dtype，动/静态shape图逻辑一样
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 */
static ge::graphStatus InferDataTypeQuantReduceScatter(gert::InferDataTypeContext *context)
{
    OPS_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(INNER_DEBUG, "Context"), return ge::GRAPH_FAILED);
    OP_LOGD(INNER_DEBUG, "start to infer datatype for quant_reduce_scatter");

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *output_dtype_ptr = attrs->GetInt(OUTPUT_DTYPE_INDEX);
    const uint64_t output_dtype = (output_dtype_ptr != nullptr ? *output_dtype_ptr : ge::DataType::DT_BF16);
    ge::DataType get_dtype = context->GetOutputDataType(OUTPUT_INDEX);
    if (output_dtype != ge::DataType::DT_BF16) {
        get_dtype = static_cast<ge::DataType>(output_dtype);
    }
    return context->SetOutputDataType(0, get_dtype);
}

IMPL_OP_INFERSHAPE(QuantReduceScatter)
    .InferShape(InferShapeQuantReduceScatter)
    .InferDataType(InferDataTypeQuantReduceScatter);

} // namespace ops
