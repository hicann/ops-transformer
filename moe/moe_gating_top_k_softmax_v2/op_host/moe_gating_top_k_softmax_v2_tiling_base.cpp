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
 * \file moe_gating_top_k_softmax_v2_tiling_base.cpp
 * \brief
 */
#include "moe_gating_top_k_softmax_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"
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
    coreNum = ascendcPlatform.GetCoreNum();
    socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::CheckOutShape(
    const gert::Shape& outShape, gert::Shape& gatingShape, bool isSoftmax, const char* tag)
{
    OP_CHECK_IF(
        (outShape.GetDimNum() != gatingShape.GetDimNum()),
        OP_LOGE(
            context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 %s and x shape num not equal, please check.", tag),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (outShape.GetDim(0) != gatingShape.GetDim(0)),
        OP_LOGE(
            context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 %s and x dim 0 not equal, please check.", tag),
        return ge::GRAPH_FAILED);
    if (gatingShape.GetDimNum() == 3U) {
        OP_CHECK_IF(
            (outShape.GetDim(1) != gatingShape.GetDim(1)),
            OP_LOGE(
                context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 %s and x dim 1 not equal, please check.", tag),
            return ge::GRAPH_FAILED);
    }
    size_t lastDimNum = gatingShape.GetDimNum() - 1;
    if (isSoftmax) {
        auto softmaxOutDtype = context_->GetOutputDesc(SOFTMAX_RESULT_INDEX)->GetDataType();
        OP_CHECK_IF(
            (softmaxOutDtype != ge::DataType::DT_FLOAT),
            OP_LOGE(
                context_->GetNodeName(),
                "Tiling4MoeGatingTopKSoftmaxV2 get softmax result type error, should be float, please check."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (outShape.GetDim(lastDimNum) != gatingShape.GetDim(lastDimNum)),
            OP_LOGE(
                context_->GetNodeName(),
                "Tiling4MoeGatingTopKSoftmaxV2 softmax and x last dim not equal, please check."),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (outShape.GetDim(lastDimNum) != k),
            OP_LOGE(
                context_->GetNodeName(),
                "Tiling4MoeGatingTopKSoftmaxV2 %s or expertIdx last dim and k not equal, please check.", tag),
            return ge::GRAPH_FAILED);
    }
    return ge::SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2BaseTiling::CheckInShape(const gert::Shape& gatingShape)
{
    OP_CHECK_IF(
        (gatingShape.GetDimNum() != 2U && gatingShape.GetDimNum() != 3U),
        OP_LOGE(
            context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 get x shape dim(=%zu) is not 2 or 3, please check.",
            gatingShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < gatingShape.GetDimNum(); i++) {
        if (i == gatingShape.GetDimNum() - 1) {
            OP_CHECK_IF(
                (gatingShape.GetDim(i) == 0),
                OP_LOGE(
                    context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 x shape last dim is zero, please check."),
                return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(
            (gatingShape.GetDim(i) > MAX_INT32),
            OP_LOGE(
                context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 x shape larger than (=%u), please check.",
                MAX_INT32),
            return ge::GRAPH_FAILED);
    }

    auto finished = context_->GetOptionalInputShape(1);
    if (finished != nullptr) {
        auto finishDtype = context_->GetOptionalInputDesc(1)->GetDataType();
        OP_CHECK_IF(
            (finishDtype != ge::DataType::DT_BOOL),
            OP_LOGE(
                context_->GetNodeName(),
                "Tiling4MoeGatingTopKSoftmaxV2 get finished type error, should be BOOL, please check."),
            return ge::GRAPH_FAILED);
        auto finishedShape = finished->GetStorageShape();
        OP_CHECK_IF(
            (finishedShape.GetDimNum() != gatingShape.GetDimNum() - 1),
            OP_LOGE(
                context_->GetNodeName(),
                "Tiling4MoeGatingTopKSoftmaxV2 get finished dim num (=%zu) error, should be (=%zu), please check.",
                finishedShape.GetDimNum(), gatingShape.GetDimNum() - 1),
            return ge::GRAPH_FAILED);
        for (size_t i = 0; i < gatingShape.GetDimNum() - 1; i++) {
            OP_CHECK_IF(
                (finishedShape.GetDim(i) != gatingShape.GetDim(i)),
                OP_LOGE(
                    context_->GetNodeName(),
                    "Tiling4MoeGatingTopKSoftmaxV2 finished and x shape not equal, please check."),
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
            OP_LOGE(
                context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 attr renorm(=%d) is wrong, please check.",
                renorm),
            return ge::GRAPH_FAILED);
    }

    const bool* softmaxFlagPtr = attrs->GetAttrPointer<bool>(2);
    softmaxFlag = 0;
    if (softmaxFlagPtr) {
        softmaxFlag = int(*softmaxFlagPtr);
    }

    if (renorm == 0 && softmaxFlag == 1) {
        OP_CHECK_IF(
            (context_->GetOutputShape(SOFTMAX_RESULT_INDEX) == nullptr),
            OP_LOGE(
                context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 softmax is nullptr, please check."),
            return ge::GRAPH_FAILED);
        auto ret = CheckOutShape(
            context_->GetOutputShape(SOFTMAX_RESULT_INDEX)->GetStorageShape(), gatingShape, true, "softmaxResult");
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
        OP_LOGE(
            context_->GetNodeName(),
            "MoeGatingTopKSoftmaxV2 x data type error, only supports float32,half,bf16. please check."),
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
    OP_CHECK_IF(
        (k <= 0 || k > int(col)),
        OP_LOGE(
            context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 attr k(=%d) is wrong, please check. col=%u", k,
            col),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (k > MAX_K),
        OP_LOGE(
            context_->GetNodeName(), "Tiling4MoeGatingTopKSoftmaxV2 attr k(=%d) is too large, please check.", k),
        return ge::GRAPH_FAILED);

    auto yDesc = context_->GetOutputDesc(OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF(
        (yDtype != dtype),
        OP_LOGE(
            context_->GetNodeName(),
            "MoeGatingTopKSoftmaxV2 y data type error, should be same with x data type. please check."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (yDtype != ge::DataType::DT_FLOAT && yDtype != ge::DataType::DT_FLOAT16 && yDtype != ge::DataType::DT_BF16),
        OP_LOGE(
            context_->GetNodeName(),
            "MoeGatingTopKSoftmaxV2 y data type error, only supports float32,half,bf16. please check."),
        return ge::GRAPH_FAILED);
    ret = CheckOutShape(context_->GetOutputShape(OUT_INDEX)->GetStorageShape(), gatingShape, false, "y");
    if (ret != ge::SUCCESS) {
        return ret;
    }

    auto expertIdxDesc = context_->GetOutputDesc(INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxDesc);
    auto expertIdxDtype = expertIdxDesc->GetDataType();
    OP_CHECK_IF(
        (expertIdxDtype != ge::DataType::DT_INT32),
        OP_LOGE(
            context_->GetNodeName(),
            "MoeGatingTopKSoftmaxV2 expertIdx data type error, only supports int32. please check."),
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