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
#include <string>
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

// 本算子的aclnn适配层为自动生成, 不会对出参做Contiguous/ViewCopy: 非连续出参会按连续布局写入
// (数据错位 + 越界写)。这里直接拒绝非连续出参。size为1的维度不参与寻址, 按torch连续性语义跳过。
static ge::graphStatus CheckOutContiguous(gert::TilingContext* context, const size_t outIndex, const char* tag)
{
    auto* outStrides = context->GetOutputStride(outIndex);
    if (outStrides == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto& outShape = context->GetOutputShape(outIndex)->GetStorageShape();
    if (outStrides->GetDimNum() != outShape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t expectedStride = 1;
    for (size_t i = outShape.GetDimNum(); i > 0; --i) {
        const size_t dim = i - 1;
        const int64_t actualStride = outStrides->GetStride(dim);
        if (outShape.GetDim(dim) != 1 && actualStride != expectedStride) {
            OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(),
                                       (std::string(tag) + " stride[" + std::to_string(dim) + "]").c_str(),
                                       std::to_string(actualStride).c_str(), std::to_string(expectedStride).c_str());
            return ge::GRAPH_PARAM_INVALID;
        }
        expectedStride *= outShape.GetDim(dim);
    }
    return ge::GRAPH_SUCCESS;
}

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
    npuArch = ascendcPlatform.GetCurNpuArch();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::CheckOutShape(const gert::Shape& outShape,
                                                              const gert::Shape& gatingShape)
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

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::CheckInShape(const gert::Shape& gatingShape)
{
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
    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::CheckFinishedInput(const gert::Shape& gatingShape)
{
    auto finished = context_->GetOptionalInputShape(1);
    if (finished == nullptr) {
        return ge::SUCCESS;
    }
    hasFinished = true;
    auto finishDtype = context_->GetOptionalInputDesc(1)->GetDataType();
    OP_CHECK_IF((finishDtype != ge::DataType::DT_BOOL),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "finished", Ops::Base::ToString(finishDtype).c_str(),
                                          "BOOL"),
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
    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::CheckOutParam(const int32_t outIndex, const gert::Shape& gatingShape,
                                                              const char* tag)
{
    auto ret = CheckOutShape(context_->GetOutputShape(outIndex)->GetStorageShape(), gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }
    return CheckOutContiguous(context_, static_cast<size_t>(outIndex), tag);
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::CheckKAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "attrs",
                    "Required attrs metadata is null; context_->GetAttrs() must provide a valid object."),
                return ge::GRAPH_FAILED);
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(kPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "k"), return ge::GRAPH_FAILED);
    k = *kPtr;
    OP_CHECK_IF((k <= 0 || k > col),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "k", std::to_string(k).c_str(),
                    ("must be greater than 0 and less than or equal to " + std::to_string(col)).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (k > MAX_K),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "k", std::to_string(k).c_str(),
                                              ("must be less than or equal to " + std::to_string(MAX_K)).c_str()),
        return ge::GRAPH_FAILED);
    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::CheckOutLastDimEqualK(const int32_t outIndex, const char* tag)
{
    const auto& outShape = context_->GetOutputShape(outIndex)->GetStorageShape();
    const int64_t lastDim = static_cast<int64_t>(outShape.GetDimNum()) - 1;
    OP_CHECK_IF((outShape.GetDim(lastDim) != k),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), (std::string(tag) + " and k").c_str(),
                    (std::to_string(outShape.GetDim(lastDim)) + " and " + std::to_string(k)).c_str(),
                    (std::string("The last dim of output ") + tag + " should be the same as attr k").c_str()),
                return ge::GRAPH_FAILED);
    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxBaseTiling::GetShapeAttrsInfo()
{
    auto gating = context_->GetInputDesc(0);
    OP_CHECK_IF(gating == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    dtype = gating->GetDataType();
    auto gatingShape = context_->GetInputShape(0)->GetStorageShape();

    auto ret = CheckInShape(gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }
    ret = CheckFinishedInput(gatingShape);
    if (ret != ge::SUCCESS) {
        return ret;
    }

    ret = CheckOutParam(OUT_INDEX, gatingShape, "y");
    if (ret != ge::SUCCESS) {
        return ret;
    }
    ret = CheckOutParam(INDICES_INDEX, gatingShape, "expertIdx");
    if (ret != ge::SUCCESS) {
        return ret;
    }
    ret = CheckOutParam(SOURCE_ROW_OUT_INDEX, gatingShape, "rowIdx");
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

    ret = CheckKAttr();
    if (ret != ge::SUCCESS) {
        return ret;
    }

    ret = CheckOutLastDimEqualK(OUT_INDEX, "y");
    if (ret != ge::SUCCESS) {
        return ret;
    }
    ret = CheckOutLastDimEqualK(INDICES_INDEX, "expertIdx");
    if (ret != ge::SUCCESS) {
        return ret;
    }
    ret = CheckOutLastDimEqualK(SOURCE_ROW_OUT_INDEX, "rowIdx");
    if (ret != ge::SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
