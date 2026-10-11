/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_softmax_v2_tiling_base.cpp
 * \brief
 */
#include "moe_gating_top_k_softmax_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
static const int32_t OUT_INDEX = 0;
static const int32_t INDICES_INDEX = 1;
static const int32_t SOFTMAX_RESULT_INDEX = 2;
static const int32_t MAX_K = 1024;
static const uint32_t MAX_INT32 = 2147483647;

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::GetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNumAiv();
    socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::CheckOutShape(const gert::Shape& outShape, gert::Shape& gatingShape,
                                                                bool isSoftmax, const char* tag)
{
    OP_CHECK_IF((outShape.GetDimNum() != gatingShape.GetDimNum()),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), (std::string(tag) + " and x").c_str(),
                    (std::to_string(outShape.GetDimNum()) + " and " + std::to_string(gatingShape.GetDimNum())).c_str(),
                    "The dim number of output should be the same as input x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((outShape.GetDim(0) != gatingShape.GetDim(0)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), (std::string(tag) + " and x").c_str(),
                    (Ops::Base::ToString(outShape) + " and " + Ops::Base::ToString(gatingShape)).c_str(),
                    "The dim 0 of output should be the same as input x"),
                return ge::GRAPH_FAILED);
    if (gatingShape.GetDimNum() == 3U) {
        OP_CHECK_IF((outShape.GetDim(1) != gatingShape.GetDim(1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), (std::string(tag) + " and x").c_str(),
                        (Ops::Base::ToString(outShape) + " and " + Ops::Base::ToString(gatingShape)).c_str(),
                        "The dim 1 of output should be the same as input x"),
                    return ge::GRAPH_FAILED);
    }
    size_t lastDimNum = gatingShape.GetDimNum() - 1;
    if (isSoftmax) {
        auto softmaxOutDtype = context_->GetOutputDesc(SOFTMAX_RESULT_INDEX)->GetDataType();
        OP_CHECK_IF((softmaxOutDtype != ge::DataType::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "softmaxResult",
                                              Ops::Base::ToString(softmaxOutDtype).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((outShape.GetDim(lastDimNum) != gatingShape.GetDim(lastDimNum)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), "softmaxResult and x",
                        (Ops::Base::ToString(outShape) + " and " + Ops::Base::ToString(gatingShape)).c_str(),
                        "The last dim of softmaxResult should be the same as input x"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((outShape.GetDim(lastDimNum) != k),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), tag, Ops::Base::ToString(outShape).c_str(),
                        ("The last dim of output should be the same as attr k, k is " + std::to_string(k)).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::CheckInShape(const gert::Shape& gatingShape)
{
    OP_CHECK_IF((gatingShape.GetDimNum() != 2U && gatingShape.GetDimNum() != 3U),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                                             std::to_string(gatingShape.GetDimNum()).c_str(), "2 or 3"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < gatingShape.GetDimNum(); i++) {
        if (i == gatingShape.GetDimNum() - 1) {
            OP_CHECK_IF((gatingShape.GetDim(i) == 0),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                                              Ops::Base::ToString(gatingShape).c_str(),
                                                              "The last dim of input x should not be zero"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF((gatingShape.GetDim(i) > MAX_INT32),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "x", Ops::Base::ToString(gatingShape).c_str(),
                        ("Each dim of input x should be no greater than " + std::to_string(MAX_INT32)).c_str()),
                    return ge::GRAPH_FAILED);
    }

    auto finished = context_->GetOptionalInputShape(1);
    if (finished != nullptr) {
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

    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::CheckOptionalAttr(gert::Shape& gatingShape)
{
    auto attrs = context_->GetAttrs();
    const int64_t* renormPtr = attrs->GetAttrPointer<int64_t>(1);
    renorm = 0;
    if (renormPtr) {
        renorm = *renormPtr;
        OP_CHECK_IF(
            (renorm < 0 || renorm > 1),
            OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "renorm", std::to_string(renorm).c_str(), "0 or 1"),
            return ge::GRAPH_FAILED);
    }

    const bool* softmaxFlagPtr = attrs->GetAttrPointer<bool>(2);
    softmaxFlag = 0;
    if (softmaxFlagPtr) {
        softmaxFlag = int(*softmaxFlagPtr);
    }

    if (renorm == 0 && softmaxFlag == 1) {
        OP_CHECK_IF((context_->GetOutputShape(SOFTMAX_RESULT_INDEX) == nullptr),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "softmaxResult", "nullptr",
                        "softmaxResult should not be nullptr when renorm is 0 and softmaxFlag is true"),
                    return ge::GRAPH_FAILED);
        auto ret = CheckOutShape(context_->GetOutputShape(SOFTMAX_RESULT_INDEX)->GetStorageShape(), gatingShape, true,
                                 "softmaxResult");
        if (ret != ge::SUCCESS) {
            return ret;
        }
    }

    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::GetShapeAttrsInfo()
{
    auto gating = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gating);
    dtype = gating->GetDataType();
    OP_CHECK_IF(
        (dtype != ge::DataType::DT_FLOAT && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(dtype).c_str(),
                                  "FLOAT, FLOAT16, BF16"),
        return ge::GRAPH_FAILED);
    auto gatingShape = context_->GetInputShape(0)->GetStorageShape();
    auto ret = CheckInShape(gatingShape);
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
    OP_CHECK_IF((k <= 0 || k > int(col)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "k", std::to_string(k).c_str(),
                    ("The attr k should be greater than 0 and no greater than the last dim of input x, col is " +
                     std::to_string(col))
                        .c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((k > MAX_K),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "k", std::to_string(k).c_str(),
                    ("The attr k should be no greater than " + std::to_string(MAX_K)).c_str()),
                return ge::GRAPH_FAILED);

    auto yDesc = context_->GetOutputDesc(OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF((yDtype != dtype),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "y and x",
                    (Ops::Base::ToString(yDtype) + " and " + Ops::Base::ToString(dtype)).c_str(),
                    "The dtype of output y should be the same as input x"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (yDtype != ge::DataType::DT_FLOAT && yDtype != ge::DataType::DT_FLOAT16 && yDtype != ge::DataType::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", Ops::Base::ToString(yDtype).c_str(),
                                  "FLOAT, FLOAT16, BF16"),
        return ge::GRAPH_FAILED);
    ret = CheckOutShape(context_->GetOutputShape(OUT_INDEX)->GetStorageShape(), gatingShape, false, "y");
    if (ret != ge::SUCCESS) {
        return ret;
    }

    auto expertIdxDesc = context_->GetOutputDesc(INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxDesc);
    auto expertIdxDtype = expertIdxDesc->GetDataType();
    OP_CHECK_IF((expertIdxDtype != ge::DataType::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expertIdx",
                                          Ops::Base::ToString(expertIdxDtype).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    ret = CheckOutShape(context_->GetOutputShape(INDICES_INDEX)->GetStorageShape(), gatingShape, false, "expertIdx");
    if (ret != ge::SUCCESS) {
        return ret;
    }

    ret = CheckOptionalAttr(gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
