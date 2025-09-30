/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v3_infershape.cpp
 * \brief
 */
 
#include <sstream>
#include <string>
#include <vector>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1U;
static constexpr size_t DIM_TWO = 2U;
static constexpr int64_t NEG_ONE = static_cast<int64_t>(-1);
static constexpr int64_t NEG_TWO = static_cast<int64_t>(-2);
static constexpr int64_t MOE_INIT_ROUTING_V3_INPUT_X = 0;
static constexpr int64_t MOE_INIT_ROUTING_V3_INPUT_EXPERT_IDX = 1;
static constexpr int64_t MOE_INIT_ROUTING_V3_INPUT_SCALE = 2;
static constexpr int64_t MOE_INIT_ROUTING_V3_INPUT_OFFSET = 3;
static constexpr int64_t MOE_INIT_ROUTING_V3_ATTR_EXPERT_NUM = 2;
static constexpr int64_t MOE_INIT_ROUTING_V3_ATTR_EXPERT_TOKEN_NUM_TYPE = 4;
static constexpr int64_t MOE_INIT_ROUTING_V3_ATTR_QUANT_MODE = 6;
static constexpr int64_t MOE_INIT_ROUTING_V3_ATTR_ACTIVE_EXPERT_RANGE = 7;
static constexpr int64_t MOE_INIT_ROUTING_V3_ATTR_ROW_IDX_TYPE = 8;
static constexpr int64_t MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_X = 0;
static constexpr int64_t MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_ROW_IDX = 1;
static constexpr int64_t MOE_INIT_ROUTING_V3_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT = 2;
static constexpr int64_t MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_SCALE = 3;
static constexpr int64_t MOE_INIT_ROUTING_V3_EXPERT_END_BOUND = 10240;
static constexpr int64_t KEY_VALUE_MODE_DIM0_NUM = 2;
enum QuantMode : int8_t {
    NON_QUANT = -1,
    STATIC_QUANT = 0,
    DYNAMIC_QUANT = 1
};
enum ExpertTokenNumType : int8_t {
    CUMSUM = 0,
    COUNT = 1,
    KEY_VALUE = 2
};

static bool isSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 <= NEG_ONE || dim2 <= NEG_ONE) {
        return true;
    }
    return dim1 == dim2;
}

static ge::graphStatus GetAndCheckAttrActiveExpertRange(const gert::RuntimeAttrs *attrs, gert::InferShapeContext* context,
                                                        int64_t &expertStart, int64_t &expertEnd)
{
    OP_LOGD(context, "Begin to do GetAndCheckAttrActiveExpertRange.");
    // Check if active_expert_range size is 2 and if expert_start < expert_end
    auto activeExpertRangePtr = attrs->GetListInt(MOE_INIT_ROUTING_V3_ATTR_ACTIVE_EXPERT_RANGE);
    if (nullptr == activeExpertRangePtr) {
        OP_LOGE(context, "The active_expert_range should be list int. But it is none.");
        return ge::GRAPH_FAILED;
    }
    int64_t activeExpertRangeSize = activeExpertRangePtr->GetSize();
    if (activeExpertRangePtr->GetSize() == DIM_TWO) {
        expertStart = activeExpertRangePtr->GetData()[0];
        expertEnd = activeExpertRangePtr->GetData()[1];
        if (expertStart >= expertEnd || expertStart < 0 || expertEnd > MOE_INIT_ROUTING_V3_EXPERT_END_BOUND) {
            OP_LOGE(context,
                    "The active_expert_range should be in [0, %ld), but the active_expert_range is [%ld, %ld).",
                    MOE_INIT_ROUTING_V3_EXPERT_END_BOUND, expertStart, expertEnd);
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context, "The active_expert_range size should be 2, but its size is %ld.", activeExpertRangeSize);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrActiveExpertRange.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertNum(const gert::RuntimeAttrs *attrs, gert::InferShapeContext* context,
                                                int64_t &experNum)
{
    OP_LOGD(context, "Begin to do GetAndCheckexperNum.");
    const int64_t *experNumPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_EXPERT_NUM);
    if (nullptr == experNumPtr) {
        OP_LOGE(context, "The expert_num should not be none.");
        return ge::GRAPH_FAILED;
    }
    experNum = *experNumPtr;
    if (experNum <= 0) {
        OP_LOGE(context, "The expert_num should be greater than 0. But it is %ld.", experNum);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrExpertNum.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertTokenNumType(const gert::RuntimeAttrs *attrs, gert::InferShapeContext* context,
                                                         int64_t &experTokenNumType)
{
    OP_LOGD(context, "Begin to do GetAndCheckexperTokenNumType.");
    const int64_t *experTokenNumTypePtr =
        attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_EXPERT_TOKEN_NUM_TYPE);
    if (nullptr == experTokenNumTypePtr) {
        OP_LOGE(context, "The expert_token_num_type should not be none.");
        return ge::GRAPH_FAILED;
    }
    experTokenNumType = *experTokenNumTypePtr;
    if (experTokenNumType < ExpertTokenNumType::CUMSUM || experTokenNumType > ExpertTokenNumType::KEY_VALUE) {
        OP_LOGE(context, "The expert_token_num_type should be %d, %d or %d. But it is %ld.",
                ExpertTokenNumType::CUMSUM, ExpertTokenNumType::COUNT, ExpertTokenNumType::KEY_VALUE,
                experTokenNumType);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrExpertTokenNumType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrQuantMode(const gert::RuntimeAttrs *attrs, gert::InferShapeContext* context,
                                                int64_t &quantMode)
{
    OP_LOGD(context, "Begin to do GetAndCheckQuantMode.");
    if (nullptr == attrs) {
        OP_LOGE(context, "The RuntimeAttrs for quant_mode is none.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_QUANT_MODE);
    if (nullptr == quantModePtr) {
        OP_LOGE(context, "The quant_mode should be %d, %d or %d. But it is none.", QuantMode::NON_QUANT,
                QuantMode::STATIC_QUANT, QuantMode::DYNAMIC_QUANT);
        return ge::GRAPH_FAILED;
    }
    quantMode = *quantModePtr;
    if (quantMode < QuantMode::NON_QUANT || quantMode > QuantMode::DYNAMIC_QUANT) {
        OP_LOGE(context, "The quant_mode should be %d, %d or %d. But it is %ld.", QuantMode::NON_QUANT,
                QuantMode::STATIC_QUANT, QuantMode::DYNAMIC_QUANT, quantMode);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrRowIdxType(const gert::RuntimeAttrs *attrs, gert::InferShapeContext* context,
                                                 int64_t &rowIdxType)
{
    OP_LOGD(context, "Begin to do GetAndCheckAttrRowIdxType.");
    if (nullptr == attrs) {
        OP_LOGE(context, "The RuntimeAttrs for row_Idx_type is none.");
        return ge::GRAPH_FAILED;
    }

    const int64_t *rowIdxTypePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_ROW_IDX_TYPE);
    if (nullptr == rowIdxTypePtr) {
        OP_LOGE(context, "The row_Idx_type should be 0 or 1. But it is none.");
        return ge::GRAPH_FAILED;
    }
    rowIdxType = *rowIdxTypePtr;
    if (rowIdxType < 0 || rowIdxType > 1) {
        OP_LOGE(context, "The row_Idx_type should be 0 or 1 But it is %ld.", rowIdxType);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrRowIdxType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputScaleShape(gert::InferShapeContext *context, const gert::Shape *xShape,
                                            const gert::Shape *scaleShape, const int64_t expertStart,
                                            const int64_t expertEnd, const int64_t quantMode)
{
    // When quant_mode is STATIC_QUANT, scale cannot be none.
    OP_CHECK_IF((nullptr == scaleShape && QuantMode::STATIC_QUANT == quantMode),
                OP_LOGE(context, "The scale cannot be none when quant_mode is %ld.", quantMode),
                return ge::GRAPH_FAILED);

    // When quant_mode is NON_QUANT or DYNAMIC_QUANT, scale can be none.
    OP_CHECK_IF((nullptr == scaleShape && (QuantMode::NON_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode)),
                OP_LOGI(context, "When quant_mode is NON_QUANT or DYNAMIC_QUANT, scale can be none."),
                return ge::GRAPH_SUCCESS);
    OP_CHECK_IF((nullptr == scaleShape),
                OP_LOGE(context, "The scale cannot be none when quant_mode is %ld.", quantMode),
                return ge::GRAPH_FAILED);

    // When quant_mode is NON_QUANT and scale is not none, the dim num of scale should be 1 and the size of scale_dim_0
    // should be same as x_shape_dim_0.
    OP_CHECK_IF(QuantMode::NON_QUANT == quantMode &&
                    (scaleShape->GetDimNum() != DIM_ONE || !isSameDim(scaleShape->GetDim(0), xShape->GetDim(0))),
                OP_LOGE(context, "The shape of scale should be (%ld), current shape is (%s).", xShape->GetDim(0),
                        Ops::Base::ToString(*scaleShape).c_str()),
                return ge::GRAPH_FAILED);

    // When quant_mode is STATIC_QUANT, the scale shape should be (end-start, ) or (end-start, 1) or (end-start, h)
    int64_t activeExpertRange = expertEnd - expertStart;
    OP_CHECK_IF(QuantMode::STATIC_QUANT == quantMode &&
                    (scaleShape->GetDimNum() == DIM_ONE && !isSameDim(scaleShape->GetDim(0), activeExpertRange)),
                OP_LOGE(context, "When quant_mode=%ld, the shape of scale should be (%ld,), current shape is (%s).",
                        quantMode, activeExpertRange, Ops::Base::ToString(*scaleShape).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(QuantMode::STATIC_QUANT == quantMode &&
                    (scaleShape->GetDimNum() == DIM_TWO && !isSameDim(scaleShape->GetDim(0), activeExpertRange)),
                OP_LOGE(context, "When quant_mode=%ld, the scale_dim_0 should be %ld, but its shape is (%s).",
                        quantMode, activeExpertRange, Ops::Base::ToString(*scaleShape).c_str()),
                return ge::GRAPH_FAILED);

    // When quant_mode is DYNAMIC_QUANT and scale is not none, the scale shape should be (end-start, h).
    OP_CHECK_IF(QuantMode::DYNAMIC_QUANT == quantMode &&
                    (scaleShape->GetDimNum() == DIM_ONE && !isSameDim(scaleShape->GetDim(0), activeExpertRange)),
                OP_LOGE(context, "When quant_mode=%ld, the scale shape should be (%ld,), current shape is (%s).",
                        quantMode, activeExpertRange, Ops::Base::ToString(*scaleShape).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(QuantMode::DYNAMIC_QUANT == quantMode &&
                    (scaleShape->GetDimNum() == DIM_TWO && (!isSameDim(scaleShape->GetDim(0), activeExpertRange) ||
                                                            !isSameDim(scaleShape->GetDim(1), xShape->GetDim(1)))),
                OP_LOGE(context, "When quant_mode=%ld, the scale shape should be (%ld, %ld), but its shape is (%s).",
                        quantMode, activeExpertRange, xShape->GetDim(1), Ops::Base::ToString(*scaleShape).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputOffsetShape(gert::InferShapeContext *context, 
                                             const gert::Shape *offsetShape, const int64_t expertStart,
                                             const int64_t expertEnd, const int64_t quantMode)
{
    // The shape of offset can be none.
    if (nullptr == offsetShape) {
        return ge::GRAPH_SUCCESS;
    }

    // If the dim num of offset is 2, the offset_dim_0 should be activeExpertRange and offset_dim_1 should be 1 or
    // x_dim_1.
    int64_t activeExpertRange = expertEnd - expertStart;
    if (QuantMode::STATIC_QUANT == quantMode) {
        if (offsetShape->GetDimNum() == DIM_ONE && !isSameDim(offsetShape->GetDim(0), activeExpertRange)) {
            OP_LOGE(context, "The shape of offset should be (%ld,), current shape is (%s).",
                    activeExpertRange, Ops::Base::ToString(*offsetShape).c_str());
            return ge::GRAPH_FAILED;
        }

        if (offsetShape->GetDimNum() == DIM_TWO && !isSameDim(offsetShape->GetDim(0), activeExpertRange)) {
            OP_LOGE(context, "The offset_dim_0 should be %ld, but its shape is (%s).", activeExpertRange,
                    Ops::Base::ToString(*offsetShape).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(gert::InferShapeContext *context, const gert::Shape *xShape,
                                       const gert::Shape *expertIdxShape, const gert::Shape *scaleShape,
                                       const gert::Shape *offsetShape, const int64_t expertStart,
                                       const int64_t expertEnd, const int64_t quantMode)
{
    // Check the shape of input_x
    if (xShape->GetDimNum() == DIM_ONE) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(context, "The dynamic dim of x should be -2, current shape is %s.",
                    Ops::Base::ToString(*xShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(context, "The dim of x should be 2 or dynamic, current shape is %s.",
                Ops::Base::ToString(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t x_n = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(1);
    if (x_n < NEG_ONE || cols < NEG_ONE) {
        OP_LOGE(context, "Invalid x shape, shape is %s.", Ops::Base::ToString(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    // Check the shape of expert_idx
    if (expertIdxShape->GetDimNum() == DIM_ONE) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(context, "The dynamic dim of expert_idx should be -2, current shape is %s.",
                    Ops::Base::ToString(*expertIdxShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(context, "The dim of expert_idx should be 2 or dynamic, current shape is %s.",
                Ops::Base::ToString(*expertIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(1);
    if (expert_idx_n < NEG_ONE || expert_idx_k < NEG_ONE) {
        OP_LOGE(context, "Invalid expert_idx shape, shape is %s.",
                Ops::Base::ToString(*expertIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(x_n, expert_idx_n)) {
        OP_LOGE(context, "The first dim of x and expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }

    // Check the shape of scale
    if (CheckInputScaleShape(context, xShape, scaleShape, expertStart, expertEnd, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check the shape of offset
    if (CheckInputOffsetShape(context, offsetShape, expertStart, expertEnd, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void ShowInputShapeAndAttrInfo(gert::InferShapeContext *context, const gert::Shape *xShape,
                                      const gert::Shape *expertIdxShape, const gert::Shape *scaleShape,
                                      const gert::Shape *offsetShape, const int64_t expertStart,
                                      const int64_t expertEnd, const int64_t quantMode, const int64_t rowIdxType)
{
    // input_x and expert_idx are all required.
    OP_LOGD(context, "x shape is: %s.", Ops::Base::ToString(*xShape).c_str());
    OP_LOGD(context, "expert_idx shape is: %s.", Ops::Base::ToString(*expertIdxShape).c_str());

    // scale is optional and can be none.
    if (nullptr == scaleShape) {
        OP_LOGD(context, "scale_shape is: none.");
    } else {
        OP_LOGD(context, "scale_shape is: %s.", Ops::Base::ToString(*scaleShape).c_str());
    }

    // offset is optional and can be none.
    OP_LOGD(context, "Begin print offset_shape.");
    if (nullptr == offsetShape) {
        OP_LOGD(context, "offset_shape is: none.");
    } else {
        OP_LOGD(context, "offset_shape is: %s.", Ops::Base::ToString(*offsetShape).c_str());
    }
    OP_LOGD(context, "End print offset_shape.");

    // Attrs are all required.
    OP_LOGD(context, "active_expert_range is: [%ld, %ld).", expertStart, expertEnd);
    OP_LOGD(context, "quant_mode is: %ld.", quantMode);
    OP_LOGD(context, "row_Idx_type is: %ld.", rowIdxType);
}

static void ShowOutputShapeInfo(gert::InferShapeContext *context, const gert::Shape *expandedXShape,
                                const gert::Shape *expandedRowIdxShape,
                                const gert::Shape *expertTokenCumsumOrCountShape, const gert::Shape *expandedScaleShape)
{
    OP_LOGD(context, "expanded_x shape is: %s after infershape.",
            Ops::Base::ToString(*expandedXShape).c_str());
    OP_LOGD(context, "expanded_row_idx shape is: %s after infershape.",
            Ops::Base::ToString(*expandedRowIdxShape).c_str());
    OP_LOGD(context, "expert_token_cumsum_or_count shape is: %s after infershape.",
            Ops::Base::ToString(*expertTokenCumsumOrCountShape).c_str());
    OP_LOGD(context, "expanded_scale shape is: %s after infershape.",
            Ops::Base::ToString(*expandedScaleShape).c_str());
}

static ge::graphStatus InferShape4MoeInitRoutingV3(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do MoeInitRoutingV3Infershape.");

    // 1. Get and check attrs
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 1.1 Get and check active_expert_range attr
    int64_t expertStart = static_cast<int64_t>(-1);
    int64_t expertEnd = static_cast<int64_t>(-1);
    if (GetAndCheckAttrActiveExpertRange(attrs, context, expertStart, expertEnd) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (nullptr == attrs) {
        OP_LOGE(context, "The attrs is none.");
        return ge::GRAPH_FAILED;
    }

    // 1.2 Get and check expert_num attr
    int64_t expertNum = static_cast<int64_t>(-1);
    if (GetAndCheckAttrExpertNum(attrs, context, expertNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 1.3 Get and check expert_token_num_type attr
    int64_t expertTokenNumType = static_cast<int64_t>(-1);
    if (GetAndCheckAttrExpertTokenNumType(attrs, context, expertTokenNumType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 1.4 Get and check quant_mode attr
    int64_t quantMode = static_cast<int64_t>(-1);
    if (GetAndCheckAttrQuantMode(attrs, context, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 1.5 Get and check row_Idx_type attr
    int64_t rowIdxType = static_cast<int64_t>(-1);
    if (GetAndCheckAttrRowIdxType(attrs, context, rowIdxType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2. Get and check input shape
    // 2.1 Get and check input_x
    const gert::Shape *xShape = context->GetInputShape(MOE_INIT_ROUTING_V3_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 2.2 Get and check expert_idx
    const gert::Shape *expertIdxShape = context->GetInputShape(MOE_INIT_ROUTING_V3_INPUT_EXPERT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);

    // 2.3 Get scale shape without checking null, because scale is optional and can be none.
    const gert::Shape *scaleShape = context->GetInputShape(MOE_INIT_ROUTING_V3_INPUT_SCALE);

    // 2.4 Get offset shape without checking null, because offset is optional and can be none.
    const gert::Shape *offsetShape = context->GetInputShape(MOE_INIT_ROUTING_V3_INPUT_OFFSET);

    // Print input shape and attr info
    ShowInputShapeAndAttrInfo(context, xShape, expertIdxShape, scaleShape, offsetShape, expertStart, expertEnd,
                              quantMode, rowIdxType);

    // Check input shape
    if (CheckInputShape(context, xShape, expertIdxShape, scaleShape, offsetShape, expertStart, expertEnd, quantMode) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 3. Infer output shape
    // 3.1 Prepare output shape
    gert::Shape *expandedXShape = context->GetOutputShape(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
    gert::Shape *expandedRowIdxShape = context->GetOutputShape(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdxShape);
    gert::Shape *expertTokenCumsumOrCountShape =
        context->GetOutputShape(MOE_INIT_ROUTING_V3_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenCumsumOrCountShape);
    gert::Shape *expandedScaleShape = context->GetOutputShape(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedScaleShape);

    int64_t x_n = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(1);

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t k = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(1);

    int64_t n = x_n > expert_idx_n ? x_n : expert_idx_n;
    int64_t outNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : n * k;

    // 3.2 Set output expanded_x shape
    expandedXShape->SetDimNum(DIM_TWO);
    expandedXShape->SetDim(0U, outNum);
    expandedXShape->SetDim(DIM_ONE, cols);

    // 3.3 Set output expanded_row_idx shape
    expandedRowIdxShape->SetDimNum(DIM_ONE);
    expandedRowIdxShape->SetDim(0U, outNum);

    // 3.4 Set output expert_token_cumsum_or_count shape
    if (expertTokenNumType == ExpertTokenNumType::KEY_VALUE) {
        expertTokenCumsumOrCountShape->SetDimNum(DIM_TWO);
        expertTokenCumsumOrCountShape->SetDim(0U, expertNum);
        expertTokenCumsumOrCountShape->SetDim(DIM_ONE, KEY_VALUE_MODE_DIM0_NUM);
    } else {
        expertTokenCumsumOrCountShape->SetDimNum(DIM_ONE);
        expertTokenCumsumOrCountShape->SetDim(0U, expertEnd - expertStart);
    }

    // 3.5 Set output expanded_scale shape
    // When scale_shape=(b*s) and non-quant, or it is dynamic quant mode, the shape of expanded_scale should be (b*s*k)
    if (QuantMode::NON_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode) {
        expandedScaleShape->SetDimNum(DIM_ONE);
        expandedScaleShape->SetDim(0U, outNum);
    }

    ShowOutputShapeInfo(context, expandedXShape, expandedRowIdxShape, expertTokenCumsumOrCountShape,
                        expandedScaleShape);
    OP_LOGD(context, "End to do MoeInitRoutingV3Infershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingV3(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "Begin to do MoeInitRoutingV3InferDataType.");

    // Get and check quant_mode attr
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t quantMode = static_cast<int64_t>(-1);

    // Infer output dtype according quant_mode
    auto xDtype = context->GetInputDataType(MOE_INIT_ROUTING_V3_INPUT_X);
    if (QuantMode::NON_QUANT == quantMode) {
        context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_X, xDtype);
    } else if (QuantMode::STATIC_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode) {
        if (ge::DT_INT8 == xDtype) {
            OP_LOGE(context, "When quant_mode=%ld, xDtype cannot be int_8.", quantMode);
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_X, ge::DT_INT8);
    }
    context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT, ge::DT_INT64);
    context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_SCALE, ge::DT_FLOAT);
    OP_LOGD(context, "End to do MoeInitRoutingV3InferDataType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4MoeInitRoutingV3(gert::InferShapeRangeContext *context)
{
    OP_LOGD(context, "Begin to do MoeInitRoutingV3InferRange.");

    // Get and check the pointers of all the outputs' shape range object
    auto expanded_x = context->GetOutputShapeRange(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_x);
    auto expanded_row_idx = context->GetOutputShapeRange(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_row_idx);
    auto count = context->GetOutputShapeRange(MOE_INIT_ROUTING_V3_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, count);
    auto expanded_scale = context->GetOutputShapeRange(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_scale);

    // Print the shape ranges of the outputs before InferShapeRange
    OP_LOGD(context, "Before InferShapeRange, expanded_x->GetMin() = %s",
            Ops::Base::ToString(*(expanded_x->GetMin())).c_str());
    OP_LOGD(context, "Before InferShapeRange, expanded_x->GetMax() = %s",
            Ops::Base::ToString(*(expanded_x->GetMax())).c_str());

    OP_LOGD(context, "Before InferShapeRange, expanded_row_idx->GetMin() = %s",
            Ops::Base::ToString(*(expanded_row_idx->GetMin())).c_str());
    OP_LOGD(context, "Before InferShapeRange, expanded_row_idx->GetMax() = %s",
            Ops::Base::ToString(*(expanded_row_idx->GetMax())).c_str());

    OP_LOGD(context, "Before InferShapeRange, count->GetMin() = %s",
            Ops::Base::ToString(*(count->GetMin())).c_str());
    OP_LOGD(context, "Before InferShapeRange, count->GetMax() = %s",
            Ops::Base::ToString(*(count->GetMax())).c_str());

    OP_LOGD(context, "Before InferShapeRange, expanded_scale->GetMin() = %s",
            Ops::Base::ToString(*(expanded_scale->GetMin())).c_str());
    OP_LOGD(context, "Before InferShapeRange, expanded_scale->GetMax() = %s",
            Ops::Base::ToString(*(expanded_scale->GetMax())).c_str());

    // Set the dim num and dim of the outputs' shape range object
    if (expanded_x->GetMin() != nullptr && expanded_x->GetMax() != nullptr) {
        expanded_x->GetMin()->SetDimNum(DIM_TWO);
        expanded_x->GetMax()->SetDimNum(DIM_TWO);
        for (size_t i = 0; i < DIM_TWO; i++) {
            expanded_x->GetMin()->SetDim(i, 0);
            expanded_x->GetMax()->SetDim(i, -1);
        }
    }

    if (expanded_row_idx->GetMin() != nullptr && expanded_row_idx->GetMax() != nullptr) {
        expanded_row_idx->GetMin()->SetDimNum(DIM_ONE);
        expanded_row_idx->GetMax()->SetDimNum(DIM_ONE);
        expanded_row_idx->GetMin()->SetDim(0, 0);
        expanded_row_idx->GetMax()->SetDim(0, -1);
    }

    if (count->GetMin() != nullptr && count->GetMax() != nullptr) {
        count->GetMin()->SetDimNum(DIM_ONE);
        count->GetMax()->SetDimNum(DIM_ONE);
        count->GetMin()->SetDim(0, 0);
        count->GetMax()->SetDim(0, -1);
    }

    if (expanded_scale->GetMin() != nullptr && expanded_scale->GetMax() != nullptr) {
        expanded_scale->GetMin()->SetDimNum(DIM_ONE);
        expanded_scale->GetMax()->SetDimNum(DIM_ONE);
        expanded_scale->GetMin()->SetDim(0, 0);
        expanded_scale->GetMax()->SetDim(0, -1);
    }

    // Print the shape ranges of the outputs after InferShapeRange
    OP_LOGD(context, "After InferShapeRange, expanded_x->GetMin() = %s",
            Ops::Base::ToString(*(expanded_x->GetMin())).c_str());
    OP_LOGD(context, "After InferShapeRange, expanded_x->GetMax() = %s",
            Ops::Base::ToString(*(expanded_x->GetMax())).c_str());

    OP_LOGD(context, "After InferShapeRange, expanded_row_idx->GetMin() = %s",
            Ops::Base::ToString(*(expanded_row_idx->GetMin())).c_str());
    OP_LOGD(context, "After InferShapeRange, expanded_row_idx->GetMax() = %s",
            Ops::Base::ToString(*(expanded_row_idx->GetMax())).c_str());

    OP_LOGD(context, "After InferShapeRange, count->GetMin() = %s",
            Ops::Base::ToString(*(count->GetMin())).c_str());
    OP_LOGD(context, "After InferShapeRange, count->GetMax() = %s",
            Ops::Base::ToString(*(count->GetMax())).c_str());

    OP_LOGD(context, "After InferShapeRange, expanded_scale->GetMin() = %s",
            Ops::Base::ToString(*(expanded_scale->GetMin())).c_str());
    OP_LOGD(context, "After InferShapeRange, expanded_scale->GetMax() = %s",
            Ops::Base::ToString(*(expanded_scale->GetMax())).c_str());

    OP_LOGD(context, "End to do MoeInitRoutingV3InferRange.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingV3)
    .InferShape(InferShape4MoeInitRoutingV3)
    .InferDataType(InferDataType4MoeInitRoutingV3)
    .InferShapeRange(InferShapeRange4MoeInitRoutingV3);
} // namespace ops