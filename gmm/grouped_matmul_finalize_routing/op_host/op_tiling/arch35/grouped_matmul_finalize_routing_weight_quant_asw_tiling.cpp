/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_finalize_routing_weight_quant_asw_tiling.h"
#include "common/include/op_host/op_const_def_transformer.h"
#include "grouped_matmul_finalize_routing_weight_quant_checker.h"
#include "../../grouped_matmul_finalize_routing_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"

namespace optiling {
using GMMFRTilingTools::GetInputsInfo;
using GMMFRTilingTools::IsMxFp8Fp4;
using GMMFRTilingTools::IsMxFp8Hif4;

namespace {
constexpr uint64_t BASIC_BLOCK_SIZE_240 = 240;
constexpr uint64_t STEP_K_4 = 4;
constexpr uint32_t SCALE_FACTOR_FP8_HIF4 = 4;
constexpr uint32_t SCALE_FACTOR_FP8_FP4 = 8;
constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
} // namespace

ge::graphStatus GMMFRWeightQuantAswTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    frInfo_ = GMMFRExtraInfo{};
    inputParams_.opName = context_->GetNodeName();

    OP_CHECK_IF(GetInputsInfo(context_, inputParams_, frInfo_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "GetInputsInfo failed."), return ge::GRAPH_FAILED);

    SetQuantMode();

    auto compileInfoPtr = context_->GetCompileInfo<GroupedMatmulFinalizeRoutingCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "CompileInfo is null."), return ge::GRAPH_FAILED);
    OP_LOGI(context_, "GetShapeAttrsInfo: npuArch(%u), aicNum(%lu), aivNum(%lu)",
            static_cast<uint32_t>(compileInfoPtr->npuArch), compileInfoPtr->aicNum, compileInfoPtr->aivNum);
    GMMFinalizeRoutingWeightQuantChecker::GMMFinalizeRoutingWeightQuantChecker checker(context_, inputParams_, frInfo_);
    return checker.CheckGroupedMatmulFinalizeRoutingWeightQuant(compileInfoPtr->npuArch, compileInfoPtr->aicNum,
                                                                compileInfoPtr->aivNum);
}

void GMMFRWeightQuantAswTiling::SetQuantMode()
{
    inputParams_.aQuantMode = QuantMode::MX_PERGROUP_MODE;
    inputParams_.bQuantMode = inputParams_.bDtype == Ops::Transformer::DT_HIFLOAT4 ? QuantMode::HIF4_PERGROUP_MODE :
                                                                                     QuantMode::MX_PERGROUP_MODE;
}

ge::graphStatus GMMFRWeightQuantAswTiling::GetPlatformInfo()
{
    auto compileInfoPtr = context_->GetCompileInfo<GroupedMatmulFinalizeRoutingCompileInfo>();
    aicoreParams_.aicNum = compileInfoPtr->aicNum;
    OP_LOGI(context_, "Compile info: aicNum(%lu)", aicoreParams_.aicNum);
    return ge::GRAPH_SUCCESS;
}

bool GMMFRWeightQuantAswTiling::IsCapable()
{
    return true;
}

ge::graphStatus GMMFRWeightQuantAswTiling::DoOpTiling()
{
    tilingData_.gmmFinalizeRoutingDataParams.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmFinalizeRoutingDataParams.batch = static_cast<uint32_t>(frInfo_.outputBs);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset = static_cast<uint32_t>(frInfo_.sharedInputOffset);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen = static_cast<uint32_t>(frInfo_.sharedInputLen);
    tilingData_.gmmFinalizeRoutingDataParams.residualScale = frInfo_.sharedInputWeight;
    tilingData_.gmmFinalizeRoutingDataParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    tilingData_.gmmFinalizeRoutingDataParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmFinalizeRoutingDataParams.hasBias = inputParams_.hasBias ? 1 : 0;
    OP_LOGD(context_,
            "DoOpTiling: groupNum(%u), batch(%u), sharedInputOffset(%u), sharedInputLen(%u), "
            "residualScale(%f), aQuantMode(%u), bQuantMode(%u), biasDtype(%u), groupListType(%u), hasBias(%u)",
            tilingData_.gmmFinalizeRoutingDataParams.groupNum, tilingData_.gmmFinalizeRoutingDataParams.batch,
            tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset,
            tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen,
            tilingData_.gmmFinalizeRoutingDataParams.residualScale, tilingData_.gmmFinalizeRoutingDataParams.aQuantMode,
            tilingData_.gmmFinalizeRoutingDataParams.bQuantMode, tilingData_.gmmFinalizeRoutingDataParams.biasDtype,
            tilingData_.gmmFinalizeRoutingDataParams.groupListType, tilingData_.gmmFinalizeRoutingDataParams.hasBias);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantAswTiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.M = inputParams_.mSize;
    tilingData_.matmulTiling.N = inputParams_.nSize;
    tilingData_.matmulTiling.Ka = inputParams_.kSize;
    tilingData_.matmulTiling.Kb = inputParams_.kSize;
    tilingData_.matmulTiling.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.matmulTiling.baseM = GmmConstant::BASIC_BLOCK_SIZE_256;
    if (IsMxFp8Fp4(inputParams_) && tilingData_.matmulTiling.isBias) {
        tilingData_.matmulTiling.baseM = BASIC_BLOCK_SIZE_240;
    }
    tilingData_.matmulTiling.baseN = GmmConstant::BASIC_BLOCK_SIZE_256;
    tilingData_.matmulTiling.baseK = GmmConstant::BASIC_BLOCK_SIZE_128;
    tilingData_.matmulTiling.stepKa = STEP_K_4;
    tilingData_.matmulTiling.stepKb = STEP_K_4;
    tilingData_.matmulTiling.dbL0C = 1;

    if (IsMxFp8Fp4(inputParams_)) {
        tilingData_.matmulTiling.mxTypePara = (GmmConstant::SCALER_FACTOR_DEFAULT << GmmConstant::SCALER_FACTOR_N_BIT) +
                                              (GmmConstant::SCALER_FACTOR_DEFAULT << GmmConstant::SCALER_FACTOR_M_BIT) +
                                              (SCALE_FACTOR_FP8_FP4 << GmmConstant::SCALER_FACTOR_B_BIT) +
                                              SCALE_FACTOR_FP8_FP4;
    } else if (IsMxFp8Hif4(inputParams_)) {
        tilingData_.matmulTiling.mxTypePara = (GmmConstant::SCALER_FACTOR_DEFAULT << GmmConstant::SCALER_FACTOR_N_BIT) +
                                              (GmmConstant::SCALER_FACTOR_DEFAULT << GmmConstant::SCALER_FACTOR_M_BIT) +
                                              (SCALE_FACTOR_FP8_HIF4 << GmmConstant::SCALER_FACTOR_B_BIT) +
                                              SCALE_FACTOR_FP8_HIF4;
    }
    OP_LOGD(context_,
            "DoLibApiTiling: M(%d), N(%d), Ka(%d), Kb(%d), usedCoreNum(%d), isBias(%u), "
            "baseM(%d), baseN(%d), baseK(%d), stepKa(%d), stepKb(%d), dbL0C(%u), mxTypePara(%d)",
            tilingData_.matmulTiling.M, tilingData_.matmulTiling.N, tilingData_.matmulTiling.Ka,
            tilingData_.matmulTiling.Kb, tilingData_.matmulTiling.usedCoreNum, tilingData_.matmulTiling.isBias,
            tilingData_.matmulTiling.baseM, tilingData_.matmulTiling.baseN, tilingData_.matmulTiling.baseK,
            tilingData_.matmulTiling.stepKa, tilingData_.matmulTiling.stepKb, tilingData_.matmulTiling.dbL0C,
            tilingData_.matmulTiling.mxTypePara);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantAswTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_, "workspaces is nullptr."), return ge::GRAPH_FAILED);
    workspaces[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantAswTiling::PostTiling()
{
    auto tilingDataSize = sizeof(GMMFinalizeRoutingTilingData);
    context_->SetBlockDim(aicoreParams_.aicNum);
    context_->SetScheduleMode(1);
    OP_CHECK_IF(tilingDataSize % sizeof(uint64_t) != 0,
                OP_LOGE(context_, "Tiling data size[%zu] is not aligned to 8", tilingDataSize),
                return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_, "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

uint64_t GMMFRWeightQuantAswTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              static_cast<uint64_t>(frInfo_.scaleType), static_cast<uint64_t>(frInfo_.rowIndexType));
}

void GMMFRWeightQuantAswTiling::Reset()
{
    inputParams_.Reset();
    frInfo_ = GMMFRExtraInfo{};
    tilingData_ = GMMFinalizeRoutingTilingData();
    const auto* rawTiling = context_ != nullptr ? context_->GetRawTilingData() : nullptr;
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr) {
        return;
    }
    OP_CHECK_IF(
        memset_s(const_cast<void*>(rawTiling->GetData()), rawTiling->GetCapacity(), 0, rawTiling->GetCapacity()) != EOK,
        OP_LOGE(context_, "Fail to clear tiling data."), return);
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulFinalizeRouting, GMMFRWeightQuantAswTiling, GMMFR_WEIGHT_QUANT_ASW_TILING);

} // namespace optiling
