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
 * \file moe_init_routing_v4_infershape.cpp
 * \brief
 */

#include <sstream>
#include <string>
#include <vector>
#include <set>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1U;
static constexpr size_t DIM_TWO = 2U;
static constexpr size_t DIM_THREE = 3U;
static constexpr int64_t NEG_ONE = static_cast<int64_t>(-1);
static constexpr int64_t NEG_TWO = static_cast<int64_t>(-2);
static constexpr int64_t MOE_INIT_ROUTING_V4_INPUT_X = 0;
static constexpr int64_t MOE_INIT_ROUTING_V4_INPUT_EXPERT_IDX = 1;
static constexpr int64_t MOE_INIT_ROUTING_V4_INPUT_SCALE = 2;
static constexpr int64_t MOE_INIT_ROUTING_V4_INPUT_OFFSET = 3;
static constexpr int64_t MOE_INIT_ROUTING_V4_INPUT_ACTIVE_NUM = 4;
static constexpr int64_t MOE_INIT_ROUTING_V4_INPUT_TOPK_WEIGHT = 5;
static constexpr int64_t MOE_INIT_ROUTING_V4_ATTR_EXPERT_CAPACITY = 0;
static constexpr int64_t MOE_INIT_ROUTING_V4_ATTR_EXPERT_NUM = 1;
static constexpr int64_t MOE_INIT_ROUTING_V4_ATTR_DROP_PAD_MODE = 2;
static constexpr int64_t MOE_INIT_ROUTING_V4_ATTR_EXPERT_TOKEN_NUM_TYPE = 3;
static constexpr int64_t MOE_INIT_ROUTING_V4_ATTR_EXPERT_TOKEN_NUM_FLAG = 4;
static constexpr int64_t MOE_INIT_ROUTING_V4_ATTR_QUANT_MODE = 5;
static constexpr int64_t MOE_INIT_ROUTING_V4_ATTR_ACTIVE_EXPERT_RANGE = 6;
static constexpr int64_t MOE_INIT_ROUTING_V4_ATTR_ROW_IDX_TYPE = 7;
static constexpr int64_t MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_X = 0;
static constexpr int64_t MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_ROW_IDX = 1;
static constexpr int64_t MOE_INIT_ROUTING_V4_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT = 2;
static constexpr int64_t MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_SCALE = 3;
static constexpr int64_t MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_TOPK_WEIGHT = 4;
static constexpr int64_t MOE_INIT_ROUTING_V4_EXPERT_END_BOUND = 10240;
static constexpr int64_t KEY_VALUE_MODE_DIM0_NUM = 2;
static constexpr int64_t MX_QUANT_BLOCK_SIZE = 32LL;
static constexpr int64_t SCALE_BLOCK_SIZE = 64LL;
static constexpr int64_t SCALE_THIRD_DIM_SIZE = 2;
static constexpr int64_t NUM_TWO = 2LL;
static constexpr int64_t FP8_PERBLOCK_BLOCK_SIZE = 128LL;
static constexpr int64_t FP8_GROUP_SIZE = 128LL;

enum DropPadMode : int8_t {
    DROP_LESS = 0,
    DROP_PAD = 1,
};

enum QuantMode : int8_t {
    NON_QUANT = -1,
    STATIC_QUANT = 0,
    DYNAMIC_QUANT = 1,
    MXQUANT_FP8_E5M2 = 2,
    MXQUANT_FP8_E4M3FN = 3,
    FP8_GROUP_E5M2 = 4,
    FP8_GROUP_E4M3FN = 5,
    HIF8_CAST = 6,
    HIF8_PERTENSOR = 7,
    HIF8_PERTOKEN = 8,
    MXQUANT_FP4_E2M1 = 9,
    FP8_PERBLOCK_E5M2 = 11,
    FP8_PERBLOCK_E4M3FN = 12,
    INT4_DYNAMIC_QUANT = 13,
    FP8_GROUP_AMAX_E5M2 = 14,
    FP8_GROUP_AMAX_E4M3FN = 15,
    MXFP8_ROUNDSCALE_AMAX_E5M2 = 16,
    MXFP8_ROUNDSCALE_AMAX_E4M3FN = 17
};

const std::set<int64_t> validQuantModes = {
    QuantMode::NON_QUANT,
    QuantMode::STATIC_QUANT,
    QuantMode::DYNAMIC_QUANT,
    QuantMode::MXQUANT_FP8_E5M2,
    QuantMode::MXQUANT_FP8_E4M3FN,
    QuantMode::FP8_GROUP_E5M2,
    QuantMode::FP8_GROUP_E4M3FN,
    QuantMode::HIF8_CAST,
    QuantMode::HIF8_PERTENSOR,
    QuantMode::HIF8_PERTOKEN,
    QuantMode::MXQUANT_FP4_E2M1,
    QuantMode::FP8_PERBLOCK_E5M2,
    QuantMode::FP8_PERBLOCK_E4M3FN,
    QuantMode::INT4_DYNAMIC_QUANT,
    QuantMode::FP8_GROUP_AMAX_E5M2,
    QuantMode::FP8_GROUP_AMAX_E4M3FN,
    QuantMode::MXFP8_ROUNDSCALE_AMAX_E5M2,
    QuantMode::MXFP8_ROUNDSCALE_AMAX_E4M3FN
};

enum ExpertTokenNumType : int8_t {
    CUMSUM = 0,
    COUNT = 1,
    KEY_VALUE = 2
};

static bool IsSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 <= NEG_ONE || dim2 <= NEG_ONE) {
        return true;
    }
    return dim1 == dim2;
}

static ge::graphStatus GetAndCheckAttrActiveExpertRange(const gert::RuntimeAttrs *attrs,
                                                        const gert::InferShapeContext *context, int64_t &expertStart,
                                                        int64_t &expertEnd, int64_t &expertNum)
{
    OP_LOGD(context, "Begin to do GetAndCheckAttrActiveExpertRange.");
    auto activeExpertRangePtr = attrs->GetListInt(MOE_INIT_ROUTING_V4_ATTR_ACTIVE_EXPERT_RANGE);
    if (nullptr == activeExpertRangePtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "active_expert_range");
        return ge::GRAPH_FAILED;
    }
    int64_t activeExpertRangeSize = activeExpertRangePtr->GetSize();
    if (activeExpertRangeSize == DIM_TWO) {
        expertStart = activeExpertRangePtr->GetData()[0];
        expertEnd = activeExpertRangePtr->GetData()[1];
        if (expertStart >= expertEnd || expertStart < 0 || expertEnd > MOE_INIT_ROUTING_V4_EXPERT_END_BOUND) {
            OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "active_expert_range",
                                      ("[" + std::to_string(expertStart) + ", " + std::to_string(expertEnd) + ")"),
                                      ("[0, " + std::to_string(MOE_INIT_ROUTING_V4_EXPERT_END_BOUND) + ")"));
            return ge::GRAPH_FAILED;
        }
    } else if (activeExpertRangeSize == 0) {
        expertStart = 0;
        expertEnd = expertNum;
    } else {
        OP_LOGE_WITH_INVALID_ATTR_SIZE(context->GetNodeName(), "active_expert_range",
                                       std::to_string(activeExpertRangeSize), "2");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrActiveExpertRange.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertCapacity(const gert::RuntimeAttrs *attrs,
                                                     const gert::InferShapeContext *context,
                                                     const gert::Shape *xShape, int64_t &expertCapacity,
                                                     const int64_t dropPadMode)
{
    OP_LOGD(context, "Begin to do GetAndCheckAttrExpertCapacity.");
    const int64_t *expertCapacityPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V4_ATTR_EXPERT_CAPACITY);
    if (nullptr == expertCapacityPtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "expert_capacity");
        return ge::GRAPH_FAILED;
    }
    expertCapacity = *expertCapacityPtr;
    if (dropPadMode == DropPadMode::DROP_PAD && xShape->GetDim(0) > 0 && expertCapacity > xShape->GetDim(0)) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "expert_capacity", std::to_string(expertCapacity),
                                  ("between 0 and " + std::to_string(xShape->GetDim(0))));
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "End to do GetAndCheckAttrExpertCapacity.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertNum(const gert::RuntimeAttrs *attrs,
                                                const gert::InferShapeContext *context,
                                                int64_t &expertNum)
{
    OP_LOGD(context, "Begin to do GetAndCheckAttrExpertNum.");
    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V4_ATTR_EXPERT_NUM);
    if (nullptr == expertNumPtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "expert_num");
        return ge::GRAPH_FAILED;
    }
    expertNum = *expertNumPtr;
    if (expertNum <= 0 || expertNum > MOE_INIT_ROUTING_V4_EXPERT_END_BOUND) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "expert_num", std::to_string(expertNum),
                                  ("in range [1, " + std::to_string(MOE_INIT_ROUTING_V4_EXPERT_END_BOUND) + "]"));
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrExpertNum.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrDropPadMode(const gert::RuntimeAttrs *attrs,
                                                  const gert::InferShapeContext *context,
                                                  int64_t &dropPadMode)
{
    OP_LOGD(context, "Begin to do GetAndCheckAttrDropPadMode.");
    const int64_t *dropPadModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V4_ATTR_DROP_PAD_MODE);
    if (nullptr == dropPadModePtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "drop_pad_mode");
        return ge::GRAPH_FAILED;
    }

    dropPadMode = *dropPadModePtr;
    if (dropPadMode < DropPadMode::DROP_LESS || dropPadMode > DropPadMode::DROP_PAD) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "drop_pad_mode", std::to_string(dropPadMode), "0 or 1");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrDropPadMode.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertTokenNumType(const gert::RuntimeAttrs *attrs,
                                                         const gert::InferShapeContext *context,
                                                         int64_t &experTokenNumType)
{
    OP_LOGD(context, "Begin to do GetAndCheckexperTokenNumType.");
    const int64_t *experTokenNumTypePtr =
        attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V4_ATTR_EXPERT_TOKEN_NUM_TYPE);
    if (nullptr == experTokenNumTypePtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "expert_token_num_type");
        return ge::GRAPH_FAILED;
    }
    experTokenNumType = *experTokenNumTypePtr;
    if (experTokenNumType < ExpertTokenNumType::CUMSUM || experTokenNumType > ExpertTokenNumType::KEY_VALUE) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "expert_token_num_type",
                                  std::to_string(experTokenNumType), "0, 1 or 2");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrExpertTokenNumType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertTokenNumFlag(const gert::RuntimeAttrs *attrs,
                                                         const gert::InferShapeContext *context,
                                                         bool &experTokenNumFlag)
{
    OP_LOGD(context, "Begin to do GetAndCheckexperTokenNumType.");
    const bool *experTokenNumFlagPtr = attrs->GetAttrPointer<bool>(MOE_INIT_ROUTING_V4_ATTR_EXPERT_TOKEN_NUM_FLAG);
    if (nullptr == experTokenNumFlagPtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "expert_token_num_flag");
        return ge::GRAPH_FAILED;
    }
    experTokenNumFlag = *experTokenNumFlagPtr;
    OP_LOGD(context, "End to do GetAndCheckAttrExpertTokenNumType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrQuantMode(const gert::RuntimeAttrs *attrs,
                                                const gert::InferShapeContext *context,
                                                int64_t &quantMode)
{
    OP_LOGD(context, "Begin to do GetAndCheckQuantMode.");
    if (nullptr == attrs) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V4_ATTR_QUANT_MODE);
    if (nullptr == quantModePtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "quant_mode");
        return ge::GRAPH_FAILED;
    }
    quantMode = *quantModePtr;
    if (validQuantModes.count(quantMode) == 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "quant_mode", std::to_string(quantMode),
                                  "-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16 or 17");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "End to do GetAndCheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrRowIdxType(const gert::RuntimeAttrs *attrs,
                                                 const gert::InferShapeContext *context,
                                                 int64_t &rowIdxType, int64_t &dropPadMode)
{
    OP_LOGD(context, "Begin to do GetAndCheckAttrRowIdxType.");
    if (nullptr == attrs) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }

    const int64_t *rowIdxTypePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V4_ATTR_ROW_IDX_TYPE);
    if (nullptr == rowIdxTypePtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "row_Idx_type");
        return ge::GRAPH_FAILED;
    }
    rowIdxType = *rowIdxTypePtr;
    if (dropPadMode == DropPadMode::DROP_PAD && rowIdxType != 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "row_Idx_type", std::to_string(rowIdxType),
                                  "0 when dropPadMode is 1");
        return ge::GRAPH_FAILED;
    }

    if (rowIdxType < 0 || rowIdxType > 1) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "row_Idx_type", std::to_string(rowIdxType), "0 or 1");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do GetAndCheckAttrRowIdxType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleShapeForNonQuant(const gert::InferShapeContext *context, const gert::Shape *xShape,
                                                  const gert::Shape *scaleShape)
{
    if (scaleShape->GetDimNum() == DIM_ONE) {
        OP_CHECK_IF(scaleShape->GetDim(0) < 0 && scaleShape->GetDim(0) != NEG_ONE && scaleShape->GetDim(0) != NEG_TWO,
                    OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale",
                                              Ops::Base::ToString(*scaleShape), "-1 or -2"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(scaleShape->GetDim(0) > 0 && !IsSameDim(scaleShape->GetDim(0), xShape->GetDim(0)),
                    OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale",
                                              Ops::Base::ToString(*scaleShape),
                                              std::to_string(xShape->GetDim(0))),
                    return ge::GRAPH_FAILED);
    } else if (scaleShape->GetDimNum() == DIM_THREE) {
        OP_CHECK_IF(scaleShape->GetDim(0) < 0 && scaleShape->GetDim(0) != NEG_ONE ||
                    scaleShape->GetDim(1) < 0 && scaleShape->GetDim(1) != NEG_ONE ||
                    scaleShape->GetDim(DIM_TWO) < 0 && scaleShape->GetDim(DIM_TWO) != NEG_ONE,
                    OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale",
                                              Ops::Base::ToString(*scaleShape), "each dim is -1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(scaleShape->GetDim(0) > 0 && xShape->GetDim(0) > 0 &&
                        !IsSameDim(scaleShape->GetDim(0), xShape->GetDim(0)),
                    OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale",
                                              Ops::Base::ToString(*scaleShape),
                                              std::to_string(xShape->GetDim(0))),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(scaleShape->GetDim(1) > 0 && xShape->GetDimNum() > 2 && xShape->GetDim(1) > 0 &&
                        !IsSameDim(scaleShape->GetDim(1),
                            Ops::Base::CeilDiv<int64_t>(xShape->GetDim(1), SCALE_BLOCK_SIZE)),
                    OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale",
                                              Ops::Base::ToString(*scaleShape),
                                              std::to_string(Ops::Base::CeilDiv<int64_t>(
                                                  xShape->GetDim(1), SCALE_BLOCK_SIZE))),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(scaleShape->GetDim(DIM_TWO) > 0 && !IsSameDim(scaleShape->GetDim(DIM_TWO),
                                                                    SCALE_THIRD_DIM_SIZE),
                    OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale",
                                              Ops::Base::ToString(*scaleShape),
                                              std::to_string(SCALE_THIRD_DIM_SIZE)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "scale", std::to_string(scaleShape->GetDimNum()),
                                     "1 or 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleShapeForStaticQuant(const gert::InferShapeContext *context,
                                                     const gert::Shape *scaleShape)
{
    if (scaleShape->GetDimNum() == DIM_ONE) {
        OP_CHECK_IF(
            scaleShape->GetDim(0) != NEG_ONE && scaleShape->GetDim(0) != NEG_TWO &&
                !IsSameDim(scaleShape->GetDim(0), DIM_ONE),
            OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale", Ops::Base::ToString(*scaleShape),
                                      "-1, -2 or 1"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "scale", std::to_string(scaleShape->GetDimNum()),
                                     "1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleShapeForDynamicQuant(const gert::InferShapeContext *context,
                                                      const gert::Shape *xShape,
                                                      const gert::Shape *scaleShape, int64_t expertStart,
                                                      int64_t expertEnd)
{
    int64_t activeExpertRange = expertEnd - expertStart;
    if (scaleShape->GetDimNum() == DIM_ONE) {
        OP_CHECK_IF(scaleShape->GetDim(0) != NEG_TWO,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "scale dim[0]",
                                              std::to_string(scaleShape->GetDim(0)), "-2"),
                    return ge::GRAPH_FAILED);
    } else if (scaleShape->GetDimNum() == DIM_TWO) {
        if (scaleShape->GetDim(0) > 0) {
            OP_CHECK_IF(
                !IsSameDim(scaleShape->GetDim(0), activeExpertRange) && !IsSameDim(scaleShape->GetDim(0), DIM_ONE),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "scale dim[0]",
                                          std::to_string(scaleShape->GetDim(0)),
                                          ("1 or " + std::to_string(activeExpertRange))),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                !IsSameDim(scaleShape->GetDim(1), xShape->GetDim(1)),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "scale dim[1]",
                                          std::to_string(scaleShape->GetDim(1)),
                                          std::to_string(xShape->GetDim(1))),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                scaleShape->GetDim(0) != NEG_ONE || (scaleShape->GetDim(1) != NEG_ONE &&
                    scaleShape->GetDim(1) != xShape->GetDim(1)),
                OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale", Ops::Base::ToString(*scaleShape),
                                          ("(-1, -1) or (-1, " + std::to_string(xShape->GetDim(1)) + ")")),
                return ge::GRAPH_FAILED);
        }
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "scale", std::to_string(scaleShape->GetDimNum()),
                                     "1(dynamic shape) or 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleShapeForHif8PerTensor(const gert::InferShapeContext *context,
                                                       const gert::Shape *scaleShape)
{
    if (scaleShape->GetDimNum() != DIM_ONE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "scale", std::to_string(scaleShape->GetDimNum()),
                                     "1");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        !IsSameDim(scaleShape->GetDim(0), DIM_ONE),
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale", Ops::Base::ToString(*scaleShape), "1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleShapeForInt4DynamicQuant(const gert::InferShapeContext *context,
                                                          const gert::Shape *xShape,
                                                          const gert::Shape *scaleShape)
{
    if (scaleShape->GetDimNum() != DIM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "scale", std::to_string(scaleShape->GetDimNum()),
                                     "2");
        return ge::GRAPH_FAILED;
    }
    if (scaleShape->GetDim(0) > 0) {
        OP_CHECK_IF(!IsSameDim(scaleShape->GetDim(0), DIM_ONE),
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "scale dim[0]",
                                              std::to_string(scaleShape->GetDim(0)), "1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!IsSameDim(scaleShape->GetDim(1), xShape->GetDim(1)),
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "scale dim[1]",
                                              std::to_string(scaleShape->GetDim(1)),
                                              std::to_string(xShape->GetDim(1))),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(scaleShape->GetDim(0) != NEG_ONE ||
                        (scaleShape->GetDim(1) != NEG_ONE && scaleShape->GetDim(1) != xShape->GetDim(1)),
                    OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "scale", Ops::Base::ToString(*scaleShape),
                                              ("(-1, -1) or (-1, " + std::to_string(xShape->GetDim(1)) + ")")),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputScaleShape(const gert::InferShapeContext *context, const gert::Shape *xShape,
                                            const gert::Shape *scaleShape, const int64_t expertStart,
                                            const int64_t expertEnd, const int64_t quantMode)
{
    OP_CHECK_IF((nullptr == scaleShape && (QuantMode::STATIC_QUANT == quantMode ||
                                           QuantMode::HIF8_PERTENSOR == quantMode)),
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "scale"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((nullptr == scaleShape &&
                 (QuantMode::NON_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode ||
                  QuantMode::MXQUANT_FP8_E5M2 == quantMode || QuantMode::MXQUANT_FP8_E4M3FN == quantMode ||
                  QuantMode::FP8_GROUP_E5M2 == quantMode || QuantMode::FP8_GROUP_E4M3FN == quantMode ||
                  QuantMode::MXFP8_ROUNDSCALE_AMAX_E5M2 == quantMode ||
                  QuantMode::MXFP8_ROUNDSCALE_AMAX_E4M3FN == quantMode ||
                  QuantMode::HIF8_CAST == quantMode || QuantMode::HIF8_PERTOKEN == quantMode ||
                  QuantMode::MXQUANT_FP4_E2M1 == quantMode ||
                  QuantMode::FP8_PERBLOCK_E5M2 == quantMode || QuantMode::FP8_PERBLOCK_E4M3FN == quantMode ||
                  QuantMode::INT4_DYNAMIC_QUANT == quantMode || QuantMode::FP8_GROUP_AMAX_E5M2 == quantMode ||
                  QuantMode::FP8_GROUP_AMAX_E4M3FN == quantMode)),
                OP_LOGI(context, "When quant_mode is %ld , scale can be none.", quantMode), return ge::GRAPH_SUCCESS);

    if (QuantMode::NON_QUANT == quantMode) {
        return CheckScaleShapeForNonQuant(context, xShape, scaleShape);
    } else if (QuantMode::STATIC_QUANT == quantMode) {
        return CheckScaleShapeForStaticQuant(context, scaleShape);
    } else if (QuantMode::DYNAMIC_QUANT == quantMode) {
        return CheckScaleShapeForDynamicQuant(context, xShape, scaleShape, expertStart, expertEnd);
    } else if (QuantMode::INT4_DYNAMIC_QUANT == quantMode) {
        return CheckScaleShapeForInt4DynamicQuant(context, xShape, scaleShape);
    } else if (QuantMode::HIF8_PERTENSOR == quantMode) {
        return CheckScaleShapeForHif8PerTensor(context, scaleShape);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputOffsetShape(const gert::InferShapeContext *context, const gert::Shape *offsetShape,
                                             const int64_t quantMode)
{
    if (quantMode != QuantMode::STATIC_QUANT) {
        return ge::GRAPH_SUCCESS;
    } else if (nullptr == offsetShape) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "offset");
        return ge::GRAPH_FAILED;
    }

    if (offsetShape->GetDimNum() != DIM_ONE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "offset", std::to_string(offsetShape->GetDimNum()),
                                     "1");
        return ge::GRAPH_FAILED;
    }
    if (offsetShape->GetDim(0) != NEG_ONE && offsetShape->GetDim(0) != NEG_TWO &&
        !IsSameDim(offsetShape->GetDim(0), DIM_ONE)) {
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "offset", Ops::Base::ToString(*offsetShape),
                                  "1, -1 or -2");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckXShape(const gert::InferShapeContext *context, const gert::Shape *xShape,
                                   int64_t &x_n, int64_t &cols)
{
    if (xShape->GetDimNum() == DIM_ONE) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "x", Ops::Base::ToString(*xShape), "-2");
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()),
                                     "2 or dynamic");
        return ge::GRAPH_FAILED;
    }

    x_n = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(0);
    cols = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(1);
    if (x_n < NEG_ONE || cols < NEG_ONE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape),
                                              "invalid x shape");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertIdxShape(const gert::InferShapeContext *context, const gert::Shape *expertIdxShape,
                                           int64_t &expert_idx_n, int64_t &expert_idx_k)
{
    if (expertIdxShape->GetDimNum() == DIM_ONE) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "expert_idx", Ops::Base::ToString(*expertIdxShape),
                                      "-2");
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "expert_idx",
                                     std::to_string(expertIdxShape->GetDimNum()), "2 or dynamic");
        return ge::GRAPH_FAILED;
    }

    expert_idx_n = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(0);
    expert_idx_k = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(1);
    if (expert_idx_n < NEG_ONE || expert_idx_k < NEG_ONE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "expert_idx",
                                              Ops::Base::ToString(*expertIdxShape),
                                              "invalid expert_idx shape");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(const gert::InferShapeContext *context, const gert::Shape *xShape,
                                       const gert::Shape *expertIdxShape, const gert::Shape *scaleShape,
                                       const gert::Shape *offsetShape, const int64_t expertStart,
                                       const int64_t expertEnd, const int64_t quantMode)
{
    int64_t x_n = 0;
    int64_t cols = 0;
    if (CheckXShape(context, xShape, x_n, cols) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t expert_idx_n = 0;
    int64_t expert_idx_k = 0;
    if (CheckExpertIdxShape(context, expertIdxShape, expert_idx_n, expert_idx_k) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (!IsSameDim(x_n, expert_idx_n)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, expert_idx",
                                               (Ops::Base::ToString(*xShape) + ", " +
                                                Ops::Base::ToString(*expertIdxShape)),
                                               "the first dim of x and expert_idx should be same");
        return ge::GRAPH_FAILED;
    }

    if (CheckInputScaleShape(context, xShape, scaleShape, expertStart, expertEnd, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputOffsetShape(context, offsetShape, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void ShowOutputShapeInfo(const gert::InferShapeContext *context, const gert::Shape *expandedXShape,
                                const gert::Shape *expandedRowIdxShape,
                                const gert::Shape *expertTokenCumsumOrCountShape, const gert::Shape *expandedScaleShape,
                                const gert::Shape *expandedTopkWeightShape)
{
    OP_LOGD(context, "expanded_x shape is: %s after infershape.", Ops::Base::ToString(*expandedXShape).c_str());
    OP_LOGD(context, "expanded_row_idx shape is: %s after infershape.",
            Ops::Base::ToString(*expandedRowIdxShape).c_str());
    OP_LOGD(context, "expert_token_cumsum_or_count shape is: %s after infershape.",
            Ops::Base::ToString(*expertTokenCumsumOrCountShape).c_str());
    OP_LOGD(context, "expanded_scale shape is: %s after infershape.", Ops::Base::ToString(*expandedScaleShape).c_str());
    if (expandedTopkWeightShape != nullptr) {
        OP_LOGD(context, "expanded_topk_weight shape is: %s after infershape.",
                Ops::Base::ToString(*expandedTopkWeightShape).c_str());
    }
    OP_LOGD(context, "End to do MoeInitRoutingV4Infershape.");
}

static ge::graphStatus GetAndValidateAttrs(const gert::InferShapeContext *context,
                                           const gert::Shape *xShape, int64_t &expertNum, int64_t &expertStart,
                                           int64_t &expertEnd, int64_t &dropPadMode,
                                           int64_t &expertCapacity, int64_t &expertTokenNumType,
                                           bool &expertTokenNumFlag, int64_t &quantMode, int64_t &rowIdxType)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    if (GetAndCheckAttrExpertNum(attrs, context, expertNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckAttrActiveExpertRange(attrs, context, expertStart, expertEnd, expertNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (nullptr == attrs) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckAttrDropPadMode(attrs, context, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckAttrExpertCapacity(attrs, context, xShape, expertCapacity, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckAttrExpertTokenNumType(attrs, context, expertTokenNumType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckAttrExpertTokenNumFlag(attrs, context, expertTokenNumFlag) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckAttrQuantMode(attrs, context, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckAttrRowIdxType(attrs, context, rowIdxType, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void CalculateOutputDims(const gert::Shape *xShape, const gert::Shape *expertIdxShape,
                                int64_t &cols, int64_t &outNum, int64_t &xOutNum)
{
    int64_t x_n = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(0);
    cols = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(1);
    int64_t expert_idx_n = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t k = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(1);
    int64_t n = x_n > expert_idx_n ? x_n : expert_idx_n;

    outNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : n * k;
    xOutNum = outNum;
}

static void SetExpandedXandRowIdxShape(gert::Shape *expandedXShape, gert::Shape *expandedRowIdxShape,
                                       int64_t dropPadMode, int64_t xOutNum, int64_t outNum, int64_t expertNum,
                                       int64_t expertCapacity, int64_t cols)
{
    if (dropPadMode == DropPadMode::DROP_LESS) {
        expandedXShape->SetDimNum(DIM_TWO);
        expandedXShape->SetDim(0U, xOutNum);
        expandedXShape->SetDim(DIM_ONE, cols);
    } else {
        expandedXShape->SetDimNum(DIM_THREE);
        expandedXShape->SetDim(0U, expertNum);
        expandedXShape->SetDim(DIM_ONE, expertCapacity);
        expandedXShape->SetDim(DIM_TWO, cols);
    }
    expandedRowIdxShape->SetDimNum(DIM_ONE);
    expandedRowIdxShape->SetDim(0U, outNum);
}

static void SetExpertTokenCumsumOrCountShape(gert::Shape *expertTokenCumsumOrCountShape, bool expertTokenNumFlag,
                                             int64_t expertTokenNumType, int64_t expertNum,
                                             int64_t expertStart, int64_t expertEnd)
{
    if (expertTokenNumFlag) {
        if (expertTokenNumType == ExpertTokenNumType::KEY_VALUE) {
            expertTokenCumsumOrCountShape->SetDimNum(DIM_TWO);
            expertTokenCumsumOrCountShape->SetDim(0U, expertNum);
            expertTokenCumsumOrCountShape->SetDim(DIM_ONE, KEY_VALUE_MODE_DIM0_NUM);
        } else {
            expertTokenCumsumOrCountShape->SetDimNum(DIM_ONE);
            expertTokenCumsumOrCountShape->SetDim(0U, expertEnd - expertStart);
        }
    }
}

static void SetExpandedScaleShape(gert::Shape *expandedScaleShape, const gert::Shape *scaleShape,
                                  int64_t quantMode, int64_t dropPadMode, int64_t xOutNum,
                                  int64_t outNum, int64_t expertNum, int64_t expertCapacity, int64_t cols)
{
    if ((QuantMode::NON_QUANT == quantMode && scaleShape && scaleShape->GetDimNum() == DIM_THREE) ||
        QuantMode::MXQUANT_FP4_E2M1 == quantMode) {
        expandedScaleShape->SetDimNum(DIM_THREE);
        expandedScaleShape->SetDim(0U, xOutNum);
        int64_t dim1 = (cols == NEG_ONE) ? NEG_ONE : Ops::Base::CeilDiv<int64_t>(cols, SCALE_BLOCK_SIZE);
        expandedScaleShape->SetDim(1U, dim1);
        expandedScaleShape->SetDim(DIM_TWO, SCALE_THIRD_DIM_SIZE);
    } else if (QuantMode::NON_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode ||
               QuantMode::INT4_DYNAMIC_QUANT == quantMode) {
        expandedScaleShape->SetDimNum(DIM_ONE);
        if (dropPadMode == DropPadMode::DROP_LESS) {
            expandedScaleShape->SetDim(0U, xOutNum);
        } else {
            expandedScaleShape->SetDim(0U, expertNum * expertCapacity);
        }
    } else if (quantMode == QuantMode::MXQUANT_FP8_E5M2 || quantMode == QuantMode::MXQUANT_FP8_E4M3FN ||
               quantMode == QuantMode::MXFP8_ROUNDSCALE_AMAX_E5M2 ||
               quantMode == QuantMode::MXFP8_ROUNDSCALE_AMAX_E4M3FN) {
        expandedScaleShape->SetDimNum(DIM_TWO);
        expandedScaleShape->SetDim(0U, outNum);
        int64_t dim1 = (cols == NEG_ONE) ? NEG_ONE :
                       Ops::Base::CeilAlign<int64_t>(Ops::Base::CeilDiv<int64_t>(cols, MX_QUANT_BLOCK_SIZE), 2LL);
        expandedScaleShape->SetDim(1U, dim1);
    } else if (QuantMode::FP8_GROUP_E5M2 == quantMode || QuantMode::FP8_GROUP_E4M3FN == quantMode ||
               QuantMode::FP8_GROUP_AMAX_E5M2 == quantMode ||
               QuantMode::FP8_GROUP_AMAX_E4M3FN == quantMode) {
        expandedScaleShape->SetDimNum(DIM_TWO);
        expandedScaleShape->SetDim(0U, outNum);
        int64_t dim1 = (cols == NEG_ONE) ? NEG_ONE : Ops::Base::CeilDiv<int64_t>(cols, FP8_GROUP_SIZE);
        expandedScaleShape->SetDim(1U, dim1);
    } else if (QuantMode::HIF8_PERTOKEN == quantMode) {
        expandedScaleShape->SetDimNum(DIM_ONE);
        expandedScaleShape->SetDim(0U, outNum);
    } else if (QuantMode::FP8_PERBLOCK_E5M2 == quantMode || QuantMode::FP8_PERBLOCK_E4M3FN == quantMode) {
        expandedScaleShape->SetDimNum(DIM_THREE);
        expandedScaleShape->SetDim(0U, outNum);
        int64_t colsAligned = (cols == NEG_ONE) ? NEG_ONE :
            Ops::Base::CeilDiv<int64_t>(cols, FP8_PERBLOCK_BLOCK_SIZE * NUM_TWO);
        expandedScaleShape->SetDim(1U, colsAligned);
        expandedScaleShape->SetDim(DIM_TWO, SCALE_THIRD_DIM_SIZE);
    }
}

static void SetExpandedTopkWeightShape(gert::Shape *expandedTopkWeightShape, const gert::Shape *topkWeightShape,
                                       int64_t dropPadMode, int64_t outNum, int64_t expertNum,
                                       int64_t expertCapacity)
{
    if (expandedTopkWeightShape != nullptr) {
        if (topkWeightShape != nullptr) {
            int64_t topkWeightOutNum = outNum;
            if (dropPadMode == DropPadMode::DROP_PAD) {
                topkWeightOutNum = expertNum * expertCapacity;
            }
            expandedTopkWeightShape->SetDimNum(DIM_TWO);
            expandedTopkWeightShape->SetDim(0U, topkWeightOutNum);
            expandedTopkWeightShape->SetDim(DIM_ONE, DIM_ONE);
        } else {
            expandedTopkWeightShape->SetDimNum(DIM_ONE);
            expandedTopkWeightShape->SetDim(0U, 0);
        }
    }
}

static ge::graphStatus InferShape4MoeInitRoutingV4(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do MoeInitRoutingV4Infershape.");

    const gert::Shape *xShape = context->GetInputShape(MOE_INIT_ROUTING_V4_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *expertIdxShape = context->GetInputShape(MOE_INIT_ROUTING_V4_INPUT_EXPERT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);
    const gert::Shape *scaleShape = context->GetOptionalInputShape(MOE_INIT_ROUTING_V4_INPUT_SCALE);
    const gert::Shape *offsetShape = context->GetOptionalInputShape(MOE_INIT_ROUTING_V4_INPUT_OFFSET);
    const gert::Shape *topkWeightShape = context->GetOptionalInputShape(MOE_INIT_ROUTING_V4_INPUT_TOPK_WEIGHT);

    int64_t expertNum = -1;
    int64_t expertStart = -1;
    int64_t expertEnd = -1;
    int64_t dropPadMode = -1;
    int64_t expertCapacity = -1;
    int64_t expertTokenNumType = -1;
    bool expertTokenNumFlag = false;
    int64_t quantMode = -1;
    int64_t rowIdxType = -1;

    if (GetAndValidateAttrs(context, xShape, expertNum, expertStart, expertEnd, dropPadMode,
                            expertCapacity, expertTokenNumType, expertTokenNumFlag, quantMode, rowIdxType) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputShape(context, xShape, expertIdxShape, scaleShape, offsetShape, expertStart, expertEnd, quantMode) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *expandedXShape = context->GetOutputShape(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
    gert::Shape *expandedRowIdxShape = context->GetOutputShape(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdxShape);
    gert::Shape *expertTokenCumsumOrCountShape =
        context->GetOutputShape(MOE_INIT_ROUTING_V4_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenCumsumOrCountShape);
    gert::Shape *expandedScaleShape = context->GetOutputShape(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedScaleShape);

    int64_t cols = 0;
    int64_t outNum = 0;
    int64_t xOutNum = 0;

    CalculateOutputDims(xShape, expertIdxShape, cols, outNum, xOutNum);

    SetExpandedXandRowIdxShape(expandedXShape, expandedRowIdxShape, dropPadMode, xOutNum, outNum, expertNum,
                               expertCapacity, cols);

    SetExpertTokenCumsumOrCountShape(expertTokenCumsumOrCountShape, expertTokenNumFlag, expertTokenNumType,
                                     expertNum, expertStart, expertEnd);

    SetExpandedScaleShape(expandedScaleShape, scaleShape, quantMode, dropPadMode, xOutNum, outNum,
                          expertNum, expertCapacity, cols);

    gert::Shape *expandedTopkWeightShape = context->GetOutputShape(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_TOPK_WEIGHT);

    SetExpandedTopkWeightShape(expandedTopkWeightShape, topkWeightShape, dropPadMode, outNum, expertNum,
                               expertCapacity);

    ShowOutputShapeInfo(context, expandedXShape, expandedRowIdxShape, expertTokenCumsumOrCountShape,
                        expandedScaleShape, expandedTopkWeightShape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputDtype(const gert::InferDataTypeContext *context, ge::DataType xDtype,
                                          int64_t quantMode)
{
    if (QuantMode::STATIC_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode) {
        if (ge::DT_INT8 == xDtype) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "xDtype", Ops::Base::ToString(xDtype),
                                                  ("xDtype cannot be int8 when quant_mode=" +
                                                   std::to_string(quantMode)));
            return ge::GRAPH_FAILED;
        }
    } else if (QuantMode::INT4_DYNAMIC_QUANT == quantMode) {
        if (xDtype != ge::DT_FLOAT && xDtype != ge::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "xDtype", Ops::Base::ToString(xDtype),
                                      "DT_FLOAT or DT_BF16");
            return ge::GRAPH_FAILED;
        }
    } else if (QuantMode::MXQUANT_FP8_E5M2 == quantMode || QuantMode::MXQUANT_FP8_E4M3FN == quantMode
        || QuantMode::MXFP8_ROUNDSCALE_AMAX_E5M2 == quantMode
        || QuantMode::MXFP8_ROUNDSCALE_AMAX_E4M3FN == quantMode
        || QuantMode::HIF8_CAST == quantMode || QuantMode::HIF8_PERTOKEN == quantMode
        || QuantMode::HIF8_PERTENSOR == quantMode || QuantMode::MXQUANT_FP4_E2M1 == quantMode ||
        QuantMode::FP8_PERBLOCK_E5M2 == quantMode || QuantMode::FP8_PERBLOCK_E4M3FN == quantMode ||
        QuantMode::FP8_GROUP_E5M2 == quantMode || QuantMode::FP8_GROUP_E4M3FN == quantMode ||
        QuantMode::FP8_GROUP_AMAX_E5M2 == quantMode || QuantMode::FP8_GROUP_AMAX_E4M3FN == quantMode) {
        if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "xDtype", Ops::Base::ToString(xDtype),
                                      "DT_FLOAT16 or DT_BF16");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::DataType DetermineOutputDtypes(const gert::InferDataTypeContext *context, ge::DataType xDtype,
                                          int64_t quantMode, ge::DataType &expandedScaleDtype)
{
    ge::DataType expandedXDtype = xDtype;
    expandedScaleDtype = context->GetOptionalInputDataType(MOE_INIT_ROUTING_V4_INPUT_SCALE);
    if (expandedScaleDtype == ge::DT_UNDEFINED) {
        expandedScaleDtype = ge::DT_FLOAT;
    }

    if (QuantMode::STATIC_QUANT == quantMode) {
        expandedXDtype = ge::DT_INT8;
    } else if (QuantMode::DYNAMIC_QUANT == quantMode) {
        auto desiredExpandedXDtype = context->GetOutputDataType(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_X);
        if (desiredExpandedXDtype == ge::DT_INT4 && (xDtype == ge::DT_FLOAT || xDtype == ge::DT_BF16)) {
            expandedXDtype = ge::DT_INT4;
        } else {
            expandedXDtype = ge::DT_INT8;
        }
    } else if (QuantMode::INT4_DYNAMIC_QUANT == quantMode) {
        expandedXDtype = ge::DT_INT4;
        expandedScaleDtype = ge::DT_FLOAT;
    } else if (QuantMode::MXQUANT_FP8_E5M2 == quantMode) {
        expandedXDtype = ge::DT_FLOAT8_E5M2;
        expandedScaleDtype = ge::DT_FLOAT8_E8M0;
    } else if (QuantMode::MXQUANT_FP8_E4M3FN == quantMode) {
        expandedXDtype = ge::DT_FLOAT8_E4M3FN;
        expandedScaleDtype = ge::DT_FLOAT8_E8M0;
    } else if (QuantMode::MXFP8_ROUNDSCALE_AMAX_E5M2 == quantMode) {
        expandedXDtype = ge::DT_FLOAT8_E5M2;
        expandedScaleDtype = ge::DT_FLOAT8_E8M0;
    } else if (QuantMode::MXFP8_ROUNDSCALE_AMAX_E4M3FN == quantMode) {
        expandedXDtype = ge::DT_FLOAT8_E4M3FN;
        expandedScaleDtype = ge::DT_FLOAT8_E8M0;
    } else if (QuantMode::FP8_GROUP_E5M2 == quantMode || QuantMode::FP8_GROUP_AMAX_E5M2 == quantMode) {
        expandedXDtype = ge::DT_FLOAT8_E5M2;
        expandedScaleDtype = ge::DT_FLOAT;
    } else if (QuantMode::FP8_GROUP_E4M3FN == quantMode || QuantMode::FP8_GROUP_AMAX_E4M3FN == quantMode) {
        expandedXDtype = ge::DT_FLOAT8_E4M3FN;
        expandedScaleDtype = ge::DT_FLOAT;
    } else if (QuantMode::HIF8_CAST == quantMode) {
        expandedXDtype = ge::DT_HIFLOAT8;
    } else if (QuantMode::MXQUANT_FP4_E2M1 == quantMode) {
        expandedXDtype = ge::DT_FLOAT4_E2M1;
        expandedScaleDtype = ge::DT_FLOAT8_E8M0;
    } else if (QuantMode::FP8_PERBLOCK_E5M2 == quantMode) {
        expandedXDtype = ge::DT_FLOAT8_E5M2;
        expandedScaleDtype = ge::DT_FLOAT;
    } else if (QuantMode::FP8_PERBLOCK_E4M3FN == quantMode) {
        expandedXDtype = ge::DT_FLOAT8_E4M3FN;
        expandedScaleDtype = ge::DT_FLOAT;
    } else if (QuantMode::NON_QUANT == quantMode &&
               (xDtype == ge::DT_FLOAT8_E5M2 || xDtype == ge::DT_FLOAT8_E4M3FN || xDtype == ge::DT_FLOAT4_E2M1)) {
        expandedScaleDtype = ge::DT_FLOAT8_E8M0;
    }

    return expandedXDtype;
}

static ge::graphStatus InferDataType4MoeInitRoutingV4(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "Begin to do MoeInitRoutingV4InferDataType.");

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V4_ATTR_QUANT_MODE);
    if (nullptr == quantModePtr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "quant_mode");
        return ge::GRAPH_FAILED;
    }
    int64_t quantMode = *quantModePtr;
    auto xDtype = context->GetInputDataType(MOE_INIT_ROUTING_V4_INPUT_X);
    if (ValidateInputDtype(context, xDtype, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto activeNumDtype = context->GetOptionalInputDataType(MOE_INIT_ROUTING_V4_INPUT_ACTIVE_NUM);
    if (activeNumDtype != ge::DT_UNDEFINED && activeNumDtype != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "active_num",
                                  Ops::Base::ToString(activeNumDtype), "DT_INT64");
        return ge::GRAPH_FAILED;
    }

    ge::DataType expandedScaleDtype = ge::DT_FLOAT;
    auto expandedXDtype = DetermineOutputDtypes(context, xDtype, quantMode, expandedScaleDtype);

    context->SetOutputDataType(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_X, expandedXDtype);
    context->SetOutputDataType(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(MOE_INIT_ROUTING_V4_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT, ge::DT_INT64);
    context->SetOutputDataType(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_SCALE, expandedScaleDtype);
    context->SetOutputDataType(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_TOPK_WEIGHT, ge::DT_FLOAT);

    OP_LOGD(context, "End to do MoeInitRoutingV4InferDataType.");
    return ge::GRAPH_SUCCESS;
}

static void SetExpandedScaleShapeRanges(gert::InferShapeRangeContext *context, int64_t quantMode)
{
    auto expanded_scale = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_SCALE);
    if (expanded_scale->GetMin() != nullptr && expanded_scale->GetMax() != nullptr) {
        size_t dimNum = DIM_ONE;
        if (quantMode == QuantMode::MXQUANT_FP4_E2M1) {
            dimNum = DIM_THREE;
        } else if (quantMode == QuantMode::MXQUANT_FP8_E5M2 || quantMode == QuantMode::MXQUANT_FP8_E4M3FN ||
                   quantMode == QuantMode::FP8_GROUP_E5M2 || quantMode == QuantMode::FP8_GROUP_E4M3FN ||
                   quantMode == QuantMode::FP8_GROUP_AMAX_E5M2 || quantMode == QuantMode::FP8_GROUP_AMAX_E4M3FN ||
                   quantMode == QuantMode::MXFP8_ROUNDSCALE_AMAX_E5M2 ||
                   quantMode == QuantMode::MXFP8_ROUNDSCALE_AMAX_E4M3FN) {
            dimNum = DIM_TWO;
        } else if (quantMode == QuantMode::NON_QUANT) {
            auto scale = context->GetInputShapeRange(MOE_INIT_ROUTING_V4_INPUT_SCALE);
            if (scale && scale->GetMin() && scale->GetMin()->GetDimNum() == DIM_THREE) {
                dimNum = DIM_THREE;
            }
        }
        expanded_scale->GetMin()->SetDimNum(dimNum);
        expanded_scale->GetMax()->SetDimNum(dimNum);
        for (size_t i = 0; i < dimNum; i++) {
            expanded_scale->GetMin()->SetDim(i, 0);
            expanded_scale->GetMax()->SetDim(i, -1);
        }
    }
}

static void LogShapeRangeInfo(gert::InferShapeRangeContext *context, const char *phase)
{
    auto expanded_x = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_X);
    auto expanded_row_idx = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_ROW_IDX);
    auto count = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT);
    auto expanded_scale = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_SCALE);

    OP_LOGD(context, "%s InferShapeRange, expanded_x->GetMin() = %s", phase,
            Ops::Base::ToString(*(expanded_x->GetMin())).c_str());
    OP_LOGD(context, "%s InferShapeRange, expanded_x->GetMax() = %s", phase,
            Ops::Base::ToString(*(expanded_x->GetMax())).c_str());
    OP_LOGD(context, "%s InferShapeRange, expanded_row_idx->GetMin() = %s", phase,
            Ops::Base::ToString(*(expanded_row_idx->GetMin())).c_str());
    OP_LOGD(context, "%s InferShapeRange, expanded_row_idx->GetMax() = %s", phase,
            Ops::Base::ToString(*(expanded_row_idx->GetMax())).c_str());
    OP_LOGD(context, "%s InferShapeRange, count->GetMin() = %s", phase,
            Ops::Base::ToString(*(count->GetMin())).c_str());
    OP_LOGD(context, "%s InferShapeRange, count->GetMax() = %s", phase,
            Ops::Base::ToString(*(count->GetMax())).c_str());
    OP_LOGD(context, "%s InferShapeRange, expanded_scale->GetMin() = %s", phase,
            Ops::Base::ToString(*(expanded_scale->GetMin())).c_str());
    OP_LOGD(context, "%s InferShapeRange, expanded_scale->GetMax() = %s", phase,
            Ops::Base::ToString(*(expanded_scale->GetMax())).c_str());

    auto expanded_topk_weight = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_TOPK_WEIGHT);
    if (expanded_topk_weight != nullptr) {
        OP_LOGD(context, "%s InferShapeRange, expanded_topk_weight->GetMin() = %s", phase,
                Ops::Base::ToString(*(expanded_topk_weight->GetMin())).c_str());
        OP_LOGD(context, "%s InferShapeRange, expanded_topk_weight->GetMax() = %s", phase,
                Ops::Base::ToString(*(expanded_topk_weight->GetMax())).c_str());
    }
}

static ge::graphStatus InferShapeRange4MoeInitRoutingV4(gert::InferShapeRangeContext *context)
{
    OP_LOGD(context, "Begin to do MoeInitRoutingV4InferRange.");

    auto expanded_x = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_x);
    auto expanded_row_idx = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_row_idx);
    auto count = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, count);
    auto expanded_scale = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, expanded_scale);

    LogShapeRangeInfo(context, "Before");

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

    const auto *attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t *quantModePtr = attrsPtr->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V4_ATTR_QUANT_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);
    int64_t quantMode = *quantModePtr;
    SetExpandedScaleShapeRanges(context, quantMode);

    auto expandedTopkWeightRange = context->GetOutputShapeRange(MOE_INIT_ROUTING_V4_OUTPUT_EXPANDED_TOPK_WEIGHT);
    if (expandedTopkWeightRange != nullptr && expandedTopkWeightRange->GetMin() != nullptr &&
        expandedTopkWeightRange->GetMax() != nullptr) {
        expandedTopkWeightRange->GetMin()->SetDimNum(DIM_TWO);
        expandedTopkWeightRange->GetMax()->SetDimNum(DIM_TWO);
        expandedTopkWeightRange->GetMin()->SetDim(0, 0);
        expandedTopkWeightRange->GetMin()->SetDim(1, 1);
        expandedTopkWeightRange->GetMax()->SetDim(0, -1);
        expandedTopkWeightRange->GetMax()->SetDim(1, 1);
    }

    LogShapeRangeInfo(context, "After");

    OP_LOGD(context, "End to do MoeInitRoutingV4InferRange.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingV4)
    .InferShape(InferShape4MoeInitRoutingV4)
    .InferDataType(InferDataType4MoeInitRoutingV4)
    .InferShapeRange(InferShapeRange4MoeInitRoutingV4);
} // namespace ops
