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
#include "mc2_comm_algo_selector.h"
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

    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const char* group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    // 判断为空或者空字符串
    OP_TILING_CHECK(group == nullptr, OP_LOGE_FOR_INVALID_VALUE(opName_, "group", "null", "not null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group[0] == '\0', OP_LOGE_FOR_INVALID_VALUE(opName_, "group", "empty", "not empty"),
                    return ge::GRAPH_FAILED);
    // 判断group是否超过127
    const char* nullTerminator = static_cast<const char*>(memchr(group, '\0', MAX_GROUP_BUFFER_SIZE));
    OP_TILING_CHECK(nullTerminator == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "group", "length > 128", "length <= 127"),
                    return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_FOR_INVALID_VALUE(opName_, "epWorldSize", "null", "not null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE_FOR_INVALID_VALUE(opName_, "sendCounts", "null", "not null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE_FOR_INVALID_VALUE(opName_, "recvCounts", "null", "not null"),
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

QuantModePair QuantGroupedMatmulAllToAllvTilingBase::GetQuantMode(const gert::TilingContext* context,
                                                                  const char* opName) const
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "attrs");
        return QuantModePair::QUANT_PAIR_ERROR;
    }
    // 获取量化模式属性（默认为0，表示非量化）
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    if (const int64_t* ptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_X_QUANT_MODE_INDEX)) {
        gmmXQuantMode = *ptr;
    }
    if (const int64_t* ptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_WEIGHT_QUANT_MODE_INDEX)) {
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

// 960dt(ascend960) CCU组网白名单校验并强制算法名(gmma2av算子族gmm/quant共用):
// 仅960dt+CCU生效, 其余机型与引擎直接放行; 查询组网flags(与算法表同一事实源)按位与匹配,
// 仅"单层纯fullmesh 1D且rank数<=16"的组网放行并强制算法名为sole[mesh];
// 组网查询失败或组网不满足条件时报错拦截(不静默降级), 需改用ai_cpu或调整组网。
bool CheckCcuFullmesh1DTopoAndForceAlgo(const char* opName, NpuArch arch, const std::string& group, uint8_t commMode,
                                        std::string& algoName)
{
    if (arch != Ops::Base::DAV_9201) {
        return true;
    }
    if (commMode != mc2tiling::A5_CCU_ENGINE) {
        return true;
    }
    Mc2Hcom::Mc2TopoInfo topoInfo;
    uint64_t topoFlags = Mc2Hcom::TOPO_FLAG_NONE;
    if (!Mc2Hcom::Mc2CommAlgoSelector::GetTopoFlags(group.c_str(), topoInfo, topoFlags)) {
        OP_LOGE(opName,
                "Failed to get group topology info for group=%s, cannot verify CCU topology constraints on "
                "ascend960dt. Please check the group topology or use ai_cpu engine.",
                group.c_str());
        return false;
    }
    constexpr uint64_t CCU_WHITELIST_FLAGS =
        Mc2Hcom::TOPO_FLAG_SINGLE_LEVEL | Mc2Hcom::TOPO_FLAG_L0_1DMESH | Mc2Hcom::TOPO_FLAG_L0_1DMESH_COVER;
    constexpr uint32_t CCU_MESH1D_MAX_RANK_SIZE = 16U;
    if ((topoFlags & CCU_WHITELIST_FLAGS) != CCU_WHITELIST_FLAGS || topoInfo.userRankSize > CCU_MESH1D_MAX_RANK_SIZE) {
        std::string reason = "CCU engine on ascend960dt requires single-level fullmesh 1D topology with rankSize "
                             "<= 16, current userRankSize=" +
                             std::to_string(topoInfo.userRankSize) +
                             ". Please check the group topology or use ai_cpu engine.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "comm_mode", "ccu", reason.c_str());
        return false;
    }
    constexpr char CCU_MESH1D_ALGO_NAME[] = "sole[mesh]";
    if (algoName != CCU_MESH1D_ALGO_NAME) {
        OP_LOGI(opName, "Force AlltoAll algoName from [%s] to [%s] for CCU engine on ascend960dt.", algoName.c_str(),
                CCU_MESH1D_ALGO_NAME);
    }
    algoName = CCU_MESH1D_ALGO_NAME;
    return true;
}
} // namespace Mc2Tiling
