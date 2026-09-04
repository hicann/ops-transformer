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
 * \file quant_grouped_mat_mul_allto_allv_tiling_base.cpp
 * \brief
 */

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_base.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace Mc2Tiling {
constexpr uint32_t MAX_GROUP_BUFFER_SIZE = 128;

// base check required
ge::graphStatus QuantGroupedMatmulAllToAllvTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    auto gmmXTensorDesc = context_->GetInputDesc(GMM_X_INDEX);
    auto gmmWeightTensorDesc = context_->GetInputDesc(GMM_WEIGHT_INDEX);
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((gmmXTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmX"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((gmmWeightTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeight"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "y"), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    // 判断为空或者空字符串
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "group", "null", "not null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group[0] == '\0', OP_LOGE_WITH_INVALID_ATTR(opName_, "group", "empty", "not empty"),
                    return ge::GRAPH_FAILED);
    // 判断group是否超过127
    const char *nullTerminator = static_cast<const char *>(memchr(group, '\0', MAX_GROUP_BUFFER_SIZE));
    OP_TILING_CHECK(nullTerminator == nullptr,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "group", "length > 128", "length <= 127"),
                    return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "epWorldSize", "null", "not null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "sendCounts", "null", "not null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "recvCounts", "null", "not null"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to get platform info"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
uint64_t QuantGroupedMatmulAllToAllvTilingBase::GetTilingKey() const
{
    return 0;
}

QuantModePair QuantGroupedMatmulAllToAllvTilingBase::GetQuantMode(const gert::TilingContext *context,
                                                                  const char *opName) const
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "attrs");
        return QuantModePair::QUANT_PAIR_ERROR;
    }
    // 获取量化模式属性（默认为0，表示非量化）
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_X_QUANT_MODE_INDEX)) {
        gmmXQuantMode = *ptr;
    }
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_WEIGHT_QUANT_MODE_INDEX)) {
        gmmWeightQuantMode = *ptr;
    }
    OP_LOGD(opName, "gmmXQuantMode=%ld, gmmWeightQuantMode=%ld", gmmXQuantMode, gmmWeightQuantMode);

    if (gmmXQuantMode == QUANT_NONE && gmmWeightQuantMode == QUANT_NONE) {
        return QUANT_PAIR_NONE;
    }

    if (gmmXQuantMode == QUANT_PERTENSOR && gmmWeightQuantMode == QUANT_PERTENSOR) {
        return QUANT_PAIR_TT;
    }

    if (gmmXQuantMode == QUANT_MX && gmmWeightQuantMode == QUANT_MX) {
        return QUANT_PAIR_MX;
    } else {
        OP_LOGD(opName, "Quantization mode error, currently gmmXQuantMode=%d, gmmWeightQuantMode=%d.", gmmXQuantMode,
                gmmWeightQuantMode);
    }
    OP_LOGD(opName, "Detected QUANT_PAIR_ERROR");
    return QUANT_PAIR_ERROR;
}
} // namespace Mc2Tiling
