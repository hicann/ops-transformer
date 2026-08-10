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
 * \file moe_init_routing_v4_tiling_arch35.cpp
 * \brief
 */
#include "moe_init_routing_v4_tiling_arch35.h"
#include "../../moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.h"
#include "../../moe_init_routing_v3/op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h"
#include "register/op_def_registry.h"
#include "log/log.h"

namespace optiling {
using Ops::Transformer::OpTiling::TilingBaseClass;

class MoeInitRoutingV4TilingArch35 : public MoeInitRoutingV3TilingArch35 {
public:
    explicit MoeInitRoutingV4TilingArch35(gert::TilingContext *context)
        : MoeInitRoutingV3TilingArch35(context)
    {
    }
    ~MoeInitRoutingV4TilingArch35() override = default;

protected:
    bool IsCapable() override
    {
        // V4 仅在 def.cpp 注册了 ascend950 配置，框架保证只有 950 才会匹配到此算子，无需再校验 SoC 版本。
        return true;
    }

    ge::graphStatus GetInputAttrsInfo() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV4TilingArch35::GetInputAttrsInfo()");

        // 当前阶段不读取 active_num 输入的值，activeNum_ 设为 -1，
        // 后续 CheckSetEmptyTensor 会将其归一化为 totalLength_（n*k）。
        // 未来动态语义阶段需通过 InputsDataDependency + GetInputTensor 读取实际值。
        activeNum_ = -1LL;

        auto attrsPtr = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);

        MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(expertCapacity_, attrsPtr, V4_ATTR_EXPERT_CAPACITY_INDEX));
        OP_LOGD(context_, "Get input attr expertCapacity = %ld.", expertCapacity_);
        MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(expertNum_, attrsPtr, V4_ATTR_EXPERT_NUM_INDEX));
        OP_LOGD(context_, "Get input attr expertNum = %ld.", expertNum_);
        MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(dropPadMode_, attrsPtr, V4_ATTR_DROP_PAD_MODE_INDEX));
        OP_LOGD(context_, "Get input attr dropPadMode = %ld.", dropPadMode_);
        MIRV3_CHECK_GE_RET(
            GetInputAttr<int64_t>(expertTokensNumType_, attrsPtr, V4_ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX));
        OP_LOGD(context_, "Get input attr expertTokensNumType = %ld.", expertTokensNumType_);
        MIRV3_CHECK_GE_RET(GetInputAttr<bool>(expertTokensNumFlag_, attrsPtr, V4_ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX));
        OP_LOGD(context_, "Get input attr expertTokensNumFlag = %d.", expertTokensNumFlag_);
        MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(quantMode_, attrsPtr, V4_ATTR_QUANT_MODE_INDEX));
        OP_LOGD(context_, "Get input attr quantMode = %ld.", quantMode_);
        const auto *aerPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(V4_ATTR_EXPERT_RANGE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, aerPtr);
        int64_t aerLen = aerPtr->GetSize();
        OP_CHECK_IF(aerLen != 2,
                    OP_LOGE_WITH_INVALID_ATTR_SIZE(context_->GetNodeName(), "active_expert_range",
                                                   std::to_string(aerLen), "2"),
                    return ge::GRAPH_FAILED);
        const int64_t *aerList = reinterpret_cast<const int64_t *>(aerPtr->GetData());
        expertStart_ = aerList[0];
        expertEnd_ = aerList[1];
        OP_LOGD(context_, "Extracted input attrs expertStart = %ld, expertEnd = %ld.", expertStart_, expertEnd_);
        MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(rowIdxType_, attrsPtr, V4_ATTR_ROW_IDX_TYPE_INDEX));
        OP_LOGD(context_, "Get input attr rowIdxType = %ld.", rowIdxType_);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetInputTensorsInfo() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV4TilingArch35::GetInputTensorsInfo()");

        MIRV3_CHECK_GE_RET(GetTensorShapeDtype<true>(xShape_, xDtype_, V4_INPUT_X_INDEX));
        inputXDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(xDtype_));
        inputXDtypeSize_ = (inputXDtypeSize_ > NUM_THOUSAND) ? 1 : inputXDtypeSize_;
        MIRV3_CHECK_GE_RET(GetTensorShapeDtype<true>(expertIdxShape_, expertIdxDtype_, V4_INPUT_EXPERT_IDX_INDEX));
        MIRV3_CHECK_GE_RET(
            GetOptionalInputShapeDtype(scaleShape_, scaleDtype_, isInputScale_, V4_INPUT_SCALE_INDEX));
        tilingDataPtr_->isInputScale = isInputScale_;
        if (isInputScale_) {
            inputScaleDTypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(scaleDtype_));
        }
        MIRV3_CHECK_GE_RET(
            GetOptionalInputShapeDtype(offsetShape_, offsetDtype_, isInputOffset_, V4_INPUT_OFFSET_INDEX));
        tilingDataPtr_->isInputOffset = isInputOffset_;
        gert::Shape activeNumShape;
        ge::DataType activeNumDtype;
        MIRV3_CHECK_GE_RET(
            GetOptionalInputShapeDtype(activeNumShape, activeNumDtype, isInputActiveNum_, V4_INPUT_ACTIVE_NUM_INDEX));
        if (isInputActiveNum_) {
            OP_CHECK_IF(activeNumDtype != ge::DT_INT64,
                        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "active_num",
                                                  Ops::Base::ToString(activeNumDtype), "DT_INT64"),
                        return ge::GRAPH_FAILED);
            auto activeNumRank = static_cast<int64_t>(activeNumShape.GetDimNum());
            OP_CHECK_IF(activeNumRank != 0 && !(activeNumRank == 1 && activeNumShape.GetDim(0) == 1),
                        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "active_num",
                                                  Ops::Base::ToString(activeNumShape), "() or (1,)"),
                        return ge::GRAPH_FAILED);
        }
        MIRV3_CHECK_GE_RET(GetOptionalInputShapeDtype(topkWeightShape_, topkWeightDtype_, isInputTopkWeight_,
                                                      V4_INPUT_TOPK_WEIGHT_INDEX));
        tilingDataPtr_->isInputTopkWeight = isInputTopkWeight_;
        OP_LOGD(context_, "Got optional input topk_weight: isInputTopkWeight = %ld.", isInputTopkWeight_);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetOutputTensorsInfo() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV4TilingArch35::GetOutputTensorsInfo()");

        MIRV3_CHECK_GE_RET(
            GetTensorShapeDtype<false>(expandedXShape_, expandedXDtype_, V4_OUTPUT_EXPANDED_X_INDEX));
        MIRV3_CHECK_GE_RET(GetTensorShapeDtype<false>(expandedRowIdxShape_, expandedRowIdxDtype_,
                                                      V4_OUTPUT_EXPANDED_ROW_IDX_INDEX));
        MIRV3_CHECK_GE_RET(GetTensorShapeDtype<false>(expertTokensCountOrCumsumShape_,
                                                      expertTokensCountOrCumsumDtype_,
                                                      V4_OUTPUT_EXPERT_TOKENS_COUNT_INDEX));
        MIRV3_CHECK_GE_RET(
            GetTensorShapeDtype<false>(expandedScaleShape_, expandedScaleDtype_, V4_OUTPUT_EXPANDED_SCALE_INDEX));
        auto expandedTopkWeightOutShapePtr = context_->GetOutputShape(V4_OUTPUT_EXPANDED_TOPK_WEIGHT_INDEX);
        if (expandedTopkWeightOutShapePtr != nullptr &&
            expandedTopkWeightOutShapePtr->GetStorageShape().GetShapeSize() > 0) {
            isOutputExpandedTopkWeight_ = 1;
            MIRV3_CHECK_GE_RET(GetTensorShapeDtype<false>(expandedTopkWeightShape_, expandedTopkWeightDtype_,
                                                          V4_OUTPUT_EXPANDED_TOPK_WEIGHT_INDEX));
        } else {
            isOutputExpandedTopkWeight_ = 0;
        }
        OP_LOGD(context_, "Got optional output expanded_topk_weight: isOutputExpandedTopkWeight = %ld.",
                isOutputExpandedTopkWeight_);

        return ge::GRAPH_SUCCESS;
    }

private:
    int64_t isInputActiveNum_ = 0;
};

REGISTER_OPS_TILING_TEMPLATE(MoeInitRoutingV4, MoeInitRoutingV4TilingArch35, 1000);

} // namespace optiling
