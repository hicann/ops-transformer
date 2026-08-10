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
 * \file moe_re_routing_v2_tiling.cpp
 * \brief
 */
#include "moe_re_routing_v2_tiling.h"
#include "../../moe_re_routing/op_host/moe_re_routing_re_tiling.h"
#include "../../moe_re_routing/op_host/moe_re_routing_r_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"

namespace optiling {

const static int64_t V2_IN_EXPERT_TOPK_WEIGHT_INDEX = 3;
const static int64_t V2_OUTPUT_PERMUTE_TOPK_WEIGHT_INDEX = 4;
const static int64_t TOPK_WEIGHT_DIM = 2;
const static int64_t TOPK_WEIGHT_DIM1_VALUE = 1;

class MoeReRoutingV2ReTiling : public MoeReRoutingReTiling {
public:
    explicit MoeReRoutingV2ReTiling(gert::TilingContext *context)
        : MoeReRoutingReTiling(context) {}
    ~MoeReRoutingV2ReTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override
    {
        ge::graphStatus ret = MoeReRoutingTilingBase::GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }

        auto topkWeightPtr = context_->GetOptionalInputDesc(V2_IN_EXPERT_TOPK_WEIGHT_INDEX);
        if (topkWeightPtr != nullptr) {
            topkWeightDtype_ = topkWeightPtr->GetDataType();
            OP_CHECK_IF(topkWeightDtype_ != ge::DT_FLOAT,
                        OP_LOGE(context_->GetNodeName(), "expert_topk_weight only supports DT_FLOAT, actual %s.",
                                ge::TypeUtils::DataTypeToSerialString(topkWeightDtype_).c_str()),
                        return ge::GRAPH_FAILED);
            hasTopkWeight_ = true;
            auto topkWeightShapePtr = context_->GetOptionalInputShape(V2_IN_EXPERT_TOPK_WEIGHT_INDEX);
            OP_CHECK_NULL_WITH_CONTEXT(context_, topkWeightShapePtr);
            auto &topkWeightShape = topkWeightShapePtr->GetStorageShape();
            OP_CHECK_IF(topkWeightShape.GetDimNum() != TOPK_WEIGHT_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            context_->GetNodeName(), "expert_topk_weight",
                            std::to_string(topkWeightShape.GetDimNum()).c_str(), "2"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(topkWeightShape.GetDim(0) != tokenSum_,
                        OP_LOGE(context_->GetNodeName(), "expert_topk_weight dim0: %ld should equal tokenSum: %ld.",
                                topkWeightShape.GetDim(0), tokenSum_),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(topkWeightShape.GetDim(1) != TOPK_WEIGHT_DIM1_VALUE,
                        OP_LOGE(context_->GetNodeName(), "expert_topk_weight dim1 must be 1, actual %ld.",
                                topkWeightShape.GetDim(1)),
                        return ge::GRAPH_FAILED);
        } else {
            auto permuteTopkWeightOutShape = context_->GetOutputShape(V2_OUTPUT_PERMUTE_TOPK_WEIGHT_INDEX);
            if (permuteTopkWeightOutShape != nullptr) {
                auto &shape = permuteTopkWeightOutShape->GetStorageShape();
                OP_CHECK_IF(shape.GetDimNum() > 0 && shape.GetDim(0) > 0,
                            OP_LOGE(context_->GetNodeName(),
                                    "permute_topk_weight output must not be provided"
                                    " when expert_topk_weight is not input."),
                            return ge::GRAPH_FAILED);
            }
        }

        return ge::GRAPH_SUCCESS;
    }
};

class MoeReRoutingV2RTiling : public MoeReRoutingRTiling {
public:
    explicit MoeReRoutingV2RTiling(gert::TilingContext *context)
        : MoeReRoutingRTiling(context) {}
    ~MoeReRoutingV2RTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override
    {
        ge::graphStatus ret = MoeReRoutingTilingBase::GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }

        auto topkWeightPtr = context_->GetOptionalInputDesc(V2_IN_EXPERT_TOPK_WEIGHT_INDEX);
        if (topkWeightPtr != nullptr) {
            topkWeightDtype_ = topkWeightPtr->GetDataType();
            OP_CHECK_IF(topkWeightDtype_ != ge::DT_FLOAT,
                        OP_LOGE(context_->GetNodeName(), "expert_topk_weight only supports DT_FLOAT, actual %s.",
                                ge::TypeUtils::DataTypeToSerialString(topkWeightDtype_).c_str()),
                        return ge::GRAPH_FAILED);
            hasTopkWeight_ = true;
            auto topkWeightShapePtr = context_->GetOptionalInputShape(V2_IN_EXPERT_TOPK_WEIGHT_INDEX);
            OP_CHECK_NULL_WITH_CONTEXT(context_, topkWeightShapePtr);
            auto &topkWeightShape = topkWeightShapePtr->GetStorageShape();
            OP_CHECK_IF(topkWeightShape.GetDimNum() != TOPK_WEIGHT_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            context_->GetNodeName(), "expert_topk_weight",
                            std::to_string(topkWeightShape.GetDimNum()).c_str(), "2"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(topkWeightShape.GetDim(0) != tokenSum_,
                        OP_LOGE(context_->GetNodeName(), "expert_topk_weight dim0: %ld should equal tokenSum: %ld.",
                                topkWeightShape.GetDim(0), tokenSum_),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(topkWeightShape.GetDim(1) != TOPK_WEIGHT_DIM1_VALUE,
                        OP_LOGE(context_->GetNodeName(), "expert_topk_weight dim1 must be 1, actual %ld.",
                                topkWeightShape.GetDim(1)),
                        return ge::GRAPH_FAILED);
        } else {
            auto permuteTopkWeightOutShape = context_->GetOutputShape(V2_OUTPUT_PERMUTE_TOPK_WEIGHT_INDEX);
            if (permuteTopkWeightOutShape != nullptr) {
                auto &shape = permuteTopkWeightOutShape->GetStorageShape();
                OP_CHECK_IF(shape.GetDimNum() > 0 && shape.GetDim(0) > 0,
                            OP_LOGE(context_->GetNodeName(),
                                    "permute_topk_weight output must not be provided"
                                    " when expert_topk_weight is not input."),
                            return ge::GRAPH_FAILED);
            }
        }

        return ge::GRAPH_SUCCESS;
    }
};

REGISTER_OPS_TILING_TEMPLATE(MoeReRoutingV2, MoeReRoutingV2ReTiling, 10000);
REGISTER_OPS_TILING_TEMPLATE(MoeReRoutingV2, MoeReRoutingV2RTiling, 11000);

}  // namespace optiling
