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
 * \file moe_gating_top_k_softmax_tiling_base.cpp
 * \brief
 */
#include "moe_gating_top_k_softmax_tiling_base.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
using namespace ge;

namespace optiling {

static const int32_t OUT_INDEX = 0;
static const int32_t INDICES_INDEX = 1;
static const int32_t SOURCE_ROW_OUT_INDEX = 2;
static const int32_t MAX_K = 1024;
static const uint32_t MAX_INT32 = 2147483647;

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::GetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNum();
    aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context_, "coreNum is: %d", coreNum);
    OP_LOGD(context_, "aivNum is: %ld", aivNum);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize = ubSizePlatForm;
    OP_LOGD(context_, "ubSize is: %lu", ubSize);
    socVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::CheckOutShape(const gert::Shape& outShape, gert::Shape& gatingShape)
{
    OP_CHECK_IF((outShape.GetDimNum() != gatingShape.GetDimNum()),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "out and x",
                    (std::to_string(outShape.GetDimNum()) + " and " + std::to_string(gatingShape.GetDimNum())).c_str(),
                    "The dim number of output should be the same as input x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((outShape.GetDim(0) != gatingShape.GetDim(0)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "out and x",
                    (Ops::Base::ToString(outShape) + " and " + Ops::Base::ToString(gatingShape)).c_str(),
                    "The dim 0 of output should be the same as input x"),
                return ge::GRAPH_FAILED);
    if (gatingShape.GetDimNum() == 3U) {
        OP_CHECK_IF((outShape.GetDim(1) != gatingShape.GetDim(1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), "out and x",
                        (Ops::Base::ToString(outShape) + " and " + Ops::Base::ToString(gatingShape)).c_str(),
                        "The dim 1 of output should be the same as input x"),
                    return ge::GRAPH_FAILED);
    }
    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::GetShapeAttrsInfo()
{
    auto gating = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gating);
    dtype = gating->GetDataType();
    auto gatingShape = context_->GetInputShape(0)->GetStorageShape();
    OP_CHECK_IF((gatingShape.GetDimNum() != 2U && gatingShape.GetDimNum() != 3U),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                                             std::to_string(gatingShape.GetDimNum()).c_str(), "2 or 3"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < gatingShape.GetDimNum(); i++) {
        OP_CHECK_IF((gatingShape.GetDim(i) == 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                                          Ops::Base::ToString(gatingShape).c_str(),
                                                          "Input x shape should not contain zero."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((gatingShape.GetDim(i) > MAX_INT32),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "x", Ops::Base::ToString(gatingShape).c_str(),
                        ("Each dim of input x should be no greater than " + std::to_string(MAX_INT32)).c_str()),
                    return ge::GRAPH_FAILED);
    }

    auto finished = context_->GetOptionalInputShape(1);
    if (finished != nullptr) {
        hasFinished = true;
        auto finishDtype = context_->GetOptionalInputDesc(1)->GetDataType();
        OP_CHECK_IF((finishDtype != ge::DataType::DT_BOOL),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "finished",
                                              Ops::Base::ToString(finishDtype).c_str(), "BOOL"),
                    return ge::GRAPH_FAILED);
        auto finishedShape = finished->GetStorageShape();
        OP_CHECK_IF((finishedShape.GetDimNum() != gatingShape.GetDimNum() - 1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "finished",
                                                 std::to_string(finishedShape.GetDimNum()).c_str(),
                                                 std::to_string(gatingShape.GetDimNum() - 1).c_str()),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < gatingShape.GetDimNum() - 1; i++) {
            OP_CHECK_IF((finishedShape.GetDim(i) != gatingShape.GetDim(i)),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context_->GetNodeName(), "finished and x",
                            (Ops::Base::ToString(finishedShape) + " and " + Ops::Base::ToString(gatingShape)).c_str(),
                            "The shape of finished should be the same as input x except the last dim"),
                        return ge::GRAPH_FAILED);
        }
    }

    auto ret = CheckOutShape(context_->GetOutputShape(OUT_INDEX)->GetStorageShape(), gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    ret = CheckOutShape(context_->GetOutputShape(INDICES_INDEX)->GetStorageShape(), gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    ret = CheckOutShape(context_->GetOutputShape(SOURCE_ROW_OUT_INDEX)->GetStorageShape(), gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    if (gatingShape.GetDimNum() == 2U) {
        row = gatingShape.GetDim(0);
        col = gatingShape.GetDim(1);
    } else {
        row = gatingShape.GetDim(0) * gatingShape.GetDim(1);
        col = gatingShape.GetDim(2U);
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kPtr);
    k = *kPtr;
    OP_CHECK_IF((k <= 0 || k > col),
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "k", std::to_string(k).c_str(),
                                          ("greater than 0 and less than or equal to " + std::to_string(col)).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((k > MAX_K),
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "k", std::to_string(k).c_str(),
                                          ("less than or equal to " + std::to_string(MAX_K)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
