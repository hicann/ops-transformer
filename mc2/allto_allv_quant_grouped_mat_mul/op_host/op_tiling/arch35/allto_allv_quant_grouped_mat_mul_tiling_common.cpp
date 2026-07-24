/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling_common.cpp
 * \brief
 */

#include "allto_allv_quant_grouped_mat_mul_tiling_common.h"
#include "mc2_comm_utils.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

constexpr int64_t CCU_MODE_RANK_THRESHOLD = 8;

ge::graphStatus AlltoAllvQuantGmmTilingCommon::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "start quant GetPlatformInfo.");
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end quant GetPlatformInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingCommon::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetShapeAttrsInfo.");
    if (GetCommonShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end GetShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingCommon::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoOpTiling.");
    if (CheckInputNotNull() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScaleFormatAndDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScaleShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivCoreNum_, aicCoreNum_, aivCoreNum_));
    context_->SetTilingKey(GetTilingKey());
    OP_TILING_CHECK(SetHcclTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "set hccl tiling failed!"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end DoOpTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingCommon::DoLibApiTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoLibApiTiling.");
    uint64_t maxMSize = 0;
    uint64_t mSize = 0;
    for (uint64_t expertIdx = 0; expertIdx < e_; expertIdx++) {
        mSize = 0;
        for (uint64_t rankIdx = 0; rankIdx < epWorldSize_; rankIdx++) {
            mSize += recvCounts[rankIdx * e_ + expertIdx];
        }
        maxMSize = std::max(mSize, maxMSize);
    }
    MC2_CHECK_LOG_RET(context_->GetNodeName(), DoGmmTiling(maxMSize));
    OP_LOGD(context_->GetNodeName(), "end DoLibApiTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingCommon::GetWorkspaceSize()
{
    OP_LOGD(context_->GetNodeName(), "start GetWorkspaceSize.");
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "can not get workspace."),
        return ge::GRAPH_FAILED);
    const uint64_t tensorListSize = 512;
    uint64_t groupListSize = sizeof(int64_t) * e_ * epWorldSize_;
    uint64_t aGroupOffsetTableSize = sizeof(uint64_t) * e_ * epWorldSize_;
    uint64_t xScaleOffsetTableSize = sizeof(uint64_t) * e_ * epWorldSize_;
    uint64_t ttScaleRepeatSize = sizeof(float) * e_ * 2;
    workspaces[0] = libApiWorkSpaceSize_ + permuteOutSize_ + permuteScaleOutSize_ +
        groupListSize + aGroupOffsetTableSize + xScaleOffsetTableSize + tensorListSize + ttScaleRepeatSize;
    OP_LOGD(context_->GetNodeName(), "end GetWorkspaceSize.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingCommon::CheckInputNotNull() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckInputNotNull.");
    // gmmX gmmWeight
    OP_TILING_CHECK((context_->GetInputDesc(GMM_X_INDEX) == nullptr),
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmX"), return ge::GRAPH_FAILED);
    // gmmWeight
    OP_TILING_CHECK(context_->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmWeight"), return ge::GRAPH_FAILED);
    // gmmY
    OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gmmY"), return ge::GRAPH_FAILED);
    // permuteOut
    if (permuteOutFlag_) {
        tilingData->isPermuteOut = true;
        OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX) == nullptr,
            OP_LOGE(context_->GetNodeName(), "When permuteOutFlag is true, output permuteOut returned null."), return ge::GRAPH_FAILED);
    }
    if (hasSharedExpertFlag_) {
        // mmX mmWeight
        OP_TILING_CHECK((context_->GetOptionalInputDesc(MM_X_INDEX) == nullptr),
            OP_LOGE(context_->GetNodeName(), "When hasSharedExpertFlag is true, input mmX can not be null."),
            return ge::GRAPH_FAILED);
        // gmmWeight
        OP_TILING_CHECK(context_->GetInputDesc(MM_WEIGHT_INDEX) == nullptr,
            OP_LOGE(context_->GetNodeName(), "When hasSharedExpertFlag is true, input mmWeight can not be null."),
            return ge::GRAPH_FAILED);
        // mmY
        OP_TILING_CHECK(context_->GetOutputDesc(OUTPUT_MM_Y_INDEX) == nullptr,
            OP_LOGE(context_->GetNodeName(), "When hasSharedExpertFlag is true, output mmy can not be null."),
                    return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckInputNotNull.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingCommon::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "start PostTiling.");
    tilingData->taskTilingInfo.BSK = bsk_;
    tilingData->taskTilingInfo.BS = bs_;
    tilingData->taskTilingInfo.H1 = h1_;
    tilingData->taskTilingInfo.H2 = h2_;
    tilingData->taskTilingInfo.A = a_;
    tilingData->taskTilingInfo.N1 = n1_;
    tilingData->taskTilingInfo.N2 = n2_;
    tilingData->taskTilingInfo.epWorldSize = epWorldSize_;
    tilingData->taskTilingInfo.e = e_;
    tilingData->taskTilingInfo.ubSize = ubSize_;
    tilingData->taskTilingInfo.aivCoreNum = aivCoreNum_;
    tilingData->taskTilingInfo.aicCoreNum = aicCoreNum_;

    uint32_t packFactor = (gmmXDataType_ == ge::DT_FLOAT4_E2M1) ? MXFP4_PACK_FACTOR : 1U;
    uint32_t expertNum = CalcExpertNum(e_, epWorldSize_, bsk_, n1_, packFactor);
    tilingData->taskTilingInfo.expertNum = expertNum;
    tilingData->taskTilingInfo.mainLoopExpertNum = expertNum;
    tilingData->taskTilingInfo.tailLoopExpertNum =
        (e_ % expertNum == 0) ? 0 : static_cast<uint32_t>(e_ % expertNum);
    tilingData->taskTilingInfo.totalLoopCount =
        static_cast<uint32_t>((e_ + expertNum - 1) / expertNum);
    tilingData->isNeedMM = hasSharedExpertFlag_;
    for (uint32_t i = 0; i < e_ * epWorldSize_; i++) {
        tilingData->taskTilingInfo.sendCnt[i] = sendCounts[i];
        tilingData->taskTilingInfo.recvCnt[i] = recvCounts[i];
    }
    PrintTaskTilingInfo(tilingData->taskTilingInfo);
    OP_LOGD(context_->GetNodeName(), "end PostTiling.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvQuantGmmTilingCommon::PrintGMMQuantTilingData(const Mc2GroupedMatmulTilingData::GMMQuantTilingData &data) const
{
    const auto &mm = data.mmTilingData;
    const auto &quantParams = data.gmmQuantParams;
    const auto &gmmArray = data.gmmArray;

    std::stringstream finalSs;
    // MM Tiling 信息
    finalSs << "MM Tiling: M=" << mm.M
            << ", N=" << mm.N
            << ", K=" << mm.Ka
            << ", usedCoreNum=" << mm.usedCoreNum
            << ", baseM=" << mm.baseM
            << ", baseN=" << mm.baseN
            << ", baseK=" << mm.baseK
            << ", singleCoreM=" << mm.singleCoreM
            << ", singleCoreN=" << mm.singleCoreN
            << ", singleCoreK=" << mm.singleCoreK
            << ", dbL0C=" << mm.dbL0C
            << ", depthA1=" << mm.depthA1
            << ", depthB1=" << mm.depthB1
            << ", stepKa=" << mm.stepKa
            << ", stepKb=" << mm.stepKb
            << ", stepM=" << mm.stepM
            << ", stepN=" << mm.stepN
            << ", iterateOrder=" << mm.iterateOrder;
    // Quant Params 信息
    finalSs << " | Quant Params: groupNum=" << quantParams.groupNum
            << ", activeType=" << quantParams.activeType
            << ", aQuantMode=" << quantParams.aQuantMode
            << ", bQuantMode=" << quantParams.bQuantMode
            << ", singleX=" << static_cast<int32_t>(quantParams.singleX)
            << ", singleW=" << static_cast<int32_t>(quantParams.singleW)
            << ", singleY=" << static_cast<int32_t>(quantParams.singleY)
            << ", groupType=" << static_cast<int32_t>(quantParams.groupType)
            << ", groupListType=" << static_cast<int32_t>(quantParams.groupListType)
            << ", hasBias=" << static_cast<int32_t>(quantParams.hasBias)
            << ", reserved=" << quantParams.reserved;
    // Array 信息
    finalSs << " | Array: mList[0]=" << gmmArray.mList[0]
            << ", kList[0]=" << gmmArray.kList[0]
            << ", nList[0]=" << gmmArray.nList[0];
    OP_LOGI(context_->GetNodeName(), "AlltoAllvQuantGmmTilingCommon MmTilingParams:\n%s", finalSs.str().c_str());
}

void AlltoAllvQuantGmmTilingCommon::PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo &taskTilingInfo) const
{
    std::stringstream ss;
    ss << "TaskTilingInfo: ";
    // TaskTilingInfo 信息
    ss << "BSK=" << taskTilingInfo.BSK
        << ", BS=" << taskTilingInfo.BS
        << ", H1=" << taskTilingInfo.H1
        << ", H2=" << taskTilingInfo.H2
        << ", A=" << taskTilingInfo.A
        << ", N1=" << taskTilingInfo.N1
        << ", N2=" << taskTilingInfo.N2;
    ss << ", epWorldSize=" << taskTilingInfo.epWorldSize
        << ", e=" << taskTilingInfo.e;
    ss << ", mainLoopExpertNum=" << taskTilingInfo.mainLoopExpertNum
        << ", tailLoopExpertNum=" << taskTilingInfo.tailLoopExpertNum
        << ", totalLoopCount=" << taskTilingInfo.totalLoopCount;
    // SendCounts 信息
    ss << "\nSendCounts: ";
    for (int i = 0; i < e_ * epWorldSize_; i++) {
        if (i != 0) {
            ss << " ,";
        }
        ss << taskTilingInfo.sendCnt[i];
    }
    // RecvCounts 信息
    ss << "\nRecvCounts: ";
    for (int i = 0; i < e_ * epWorldSize_; i++) {
        if (i != 0) {
            ss << " ,";
        }
        ss << taskTilingInfo.recvCnt[i];
    }
    OP_LOGI(context_->GetNodeName(), "%s", ss.str().c_str());
}

ge::graphStatus AlltoAllvQuantGmmTilingCommon::QuantGetAndConvertCommMode(gert::TilingContext *context,
    uint8_t &commMode) const
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Failed to get attrs."), return ge::GRAPH_FAILED);
    const char *commModeStr = attrs->GetAttrPointer<char>(ATTR_COMM_MODE);
    OP_TILING_CHECK(commModeStr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "comm_mode"), return ge::GRAPH_FAILED);
    const size_t maxLength = 7UL;
    if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
        commMode = Mc2Comm::COMM_MODE_AICPU;
    } else if (strncmp(commModeStr, "ccu", maxLength) == 0) {
        commMode = Mc2Comm::COMM_MODE_CCU;
    } else {
        OP_LOGD(context->GetNodeName(),
                "Currently, commMode only support 'ccu', 'ai_cpu', but got %s.", commModeStr);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvQuantGmmTilingCommon::SetHcclTiling() const
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    const uint32_t alltoAllvReduceType = 0u;
    mc2tiling::HcclDataType alltoallvHcclDataType = mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED;
    // mxfp4 场景使用int8类型通信，通信量减半
    if (gmmXDataType_  == ge::DataType::DT_FLOAT4_E2M1) {
        alltoallvHcclDataType = mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT8;
    } else {
        alltoallvHcclDataType = mc2tiling::ConvertGeTypeToHcclType(context_->GetNodeName(), gmmXDataType_);
    }
    OP_TILING_CHECK(alltoallvHcclDataType == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED, OP_LOGE(context_->GetNodeName(), "Ge datatype gmmXDataType(%s) "
        "is not found in HCCL_DATA_TYPE.", ge::TypeUtils::DataTypeToSerialString(gmmXDataType_).c_str()), return ge::GRAPH_FAILED);
    uint8_t alltoAllvDataType = static_cast<uint8_t>(alltoallvHcclDataType);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType, alltoAllvDataType,
        alltoAllvDataType);
    uint8_t commMode = 0;
    if (QuantGetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commMode == Mc2Comm::COMM_MODE_CCU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
        OP_LOGE(context_->GetNodeName(),
        "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.hcclInitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
        OP_LOGE(context_->GetNodeName(),
        "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.a2avCcTiling failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
