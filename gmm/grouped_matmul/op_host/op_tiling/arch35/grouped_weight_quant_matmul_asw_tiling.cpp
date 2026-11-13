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
 * \file grouped_weight_quant_matmul_asw_tiling.cpp
 * \brief
 */

#include "grouped_weight_quant_matmul_asw_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_asw/weight_quant_asw_tiling_key.h"

namespace GmmTilingTool = optiling::GMMWeightQuantTilingTool;
namespace GmmConstants = optiling::GMMWeightQuantConstants;
using GmmTilingTool::AntiQuantMode;
using GmmTilingTool::GetShapeWithDataType;
using GmmTilingTool::IsA16MxFp4;
using GmmTilingTool::IsA16MxHiF4;
using GmmTilingTool::IsMxA8HiF4;
using GmmTilingTool::IsMxA8W4;
using GroupedMatmul::CeilAlign;

namespace optiling {

bool GroupedWeightQuantMatmulAswTiling::SetTiling(gert::TilingContext* context)
{
    context_ = context;
    inputParams_.opName = context_->GetNodeType();
    OP_CHECK_IF(!GetPlatformInfo(), OP_LOGE(inputParams_.opName, "Failed to GetPlatformInfo."), return false);
    OP_CHECK_IF(!AnalyzeInputs(), OP_LOGE(inputParams_.opName, "Failed to AnalyzeInputs."), return false);
    DoLibApiTiling();
    DoOpTiling();
    OP_CHECK_IF(!GetWorkspaceSize(), OP_LOGE(inputParams_.opName, "Failed to GetWorkspaceSize."), return false);
    OP_CHECK_IF(!PostTiling(), OP_LOGE(inputParams_.opName, "Failed to PostTiling."), return false);
    PrintTilingData();
    OP_CHECK_IF(!SetTilingKey(), OP_LOGE(inputParams_.opName, "Failed to SetTilingKey."), return false);
    return true;
}

bool GroupedWeightQuantMatmulAswTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<GMMCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(inputParams_.opName, "CompileInfoPtr is null."), return false);

        inputParams_.enableMxShapeGeneralization = compileInfoPtr->npuArch == NpuArch::DAV_9201;
        aicoreParams_.aicNum = compileInfoPtr->aicNum;
        aicoreParams_.l0cSize = compileInfoPtr->l0CSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        inputParams_.enableMxShapeGeneralization = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_9201;
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_LOGI(inputParams_.opName, "Platform info: aicNum [%llu] l0cSize [%llu].", aicoreParams_.aicNum,
            aicoreParams_.l0cSize);
    return true;
}

bool GroupedWeightQuantMatmulAswTiling::AnalyzeInputs()
{
    OP_CHECK_IF(!GmmTilingTool::AnalyzeInputs(context_, inputParams_, inputNum_),
                OP_LOGE(inputParams_.opName, "Failed to AnalyzeInputs."), return false);
    // 入参校验，之后调用GetDynamicInputShape不再需要校验空指针
    OP_CHECK_IF(!GMMWeightQuantTilingChecker(context_, inputParams_, inputNum_).CheckGroupedMatmulWeightQuant(),
                OP_LOGE(inputParams_.opName, "Failed to check grouped_matmul inputs."), return false);

    SetGroupSize();
    SetMKN();
    return true;
}

void GroupedWeightQuantMatmulAswTiling::SetMKN()
{
    auto xShapePtr = context_->GetDynamicInputShape(GmmConstants::X_INDEX, 0);
    const gert::Shape& xShape = xShapePtr->GetOriginShape();
    size_t xDimNum = xShape.GetDimNum();

    auto wShapePtr = context_->GetDynamicInputShape(GmmConstants::WEIGHT_INDEX, 0);
    const gert::Shape& wShape = wShapePtr->GetOriginShape();
    size_t wDimNum = wShape.GetDimNum();

    inputParams_.mSize = inputParams_.transX ? xShape.GetDim(xDimNum - GmmConstants::LAST_FIRST_DIM_INDEX) :
                                               xShape.GetDim(xDimNum - GmmConstants::LAST_SECOND_DIM_INDEX);
    inputParams_.kSize = inputParams_.transX ? xShape.GetDim(xDimNum - GmmConstants::LAST_SECOND_DIM_INDEX) :
                                               xShape.GetDim(xDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);
    inputParams_.nSize = inputParams_.transWeight ? wShape.GetDim(wDimNum - GmmConstants::LAST_SECOND_DIM_INDEX) :
                                                    wShape.GetDim(wDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);
}

void GroupedWeightQuantMatmulAswTiling::SetGroupSize()
{
    if (IsMxA8W4(inputParams_) || IsA16MxFp4(inputParams_)) {
        inputParams_.groupSize = GmmConstants::MX_GROUP_SIZE_32;
    } else if (IsMxA8HiF4(inputParams_) || IsA16MxHiF4(inputParams_)) {
        inputParams_.groupSize = GmmConstants::MX_GROUP_SIZE_64;
    }
}

void GroupedWeightQuantMatmulAswTiling::DoOpTiling()
{
    tilingData_.gmmWeightQuantAswParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmWeightQuantAswParams.l0cSize = aicoreParams_.l0cSize / GmmConstants::KB_UNIT;
    tilingData_.gmmWeightQuantAswParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmWeightQuantAswParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmWeightQuantAswParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmWeightQuantAswParams.groupType = inputParams_.groupType;
    tilingData_.gmmWeightQuantAswParams.groupListType = inputParams_.groupListType;
    tilingData_.gmmWeightQuantAswParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    tilingData_.gmmWeightQuantAswParams.groupSize = inputParams_.groupSize;
}

void GroupedWeightQuantMatmulAswTiling::DoLibApiTiling()
{
    CalBasicBlock();
    tilingData_.mmTilingData.M = inputParams_.mSize;
    tilingData_.mmTilingData.N = inputParams_.nSize;
    tilingData_.mmTilingData.Ka = inputParams_.kSize;
    tilingData_.mmTilingData.Kb = inputParams_.kSize;
    tilingData_.mmTilingData.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.stepKa = GmmConstants::STEP_K_4;
    tilingData_.mmTilingData.stepKb = GmmConstants::STEP_K_4;
    tilingData_.mmTilingData.isBias = inputParams_.hasBias ? 1 : 0;
}

void GroupedWeightQuantMatmulAswTiling::CalBasicBlock()
{
    basicTiling_.baseM = GmmConstants::BASIC_BLOCK_SIZE_256;
    if (IsMxA8W4(inputParams_) && inputParams_.hasBias) {
        basicTiling_.baseM = GmmConstants::BASIC_BLOCK_SIZE_240;
    } else if (IsA16MxFp4(inputParams_)) {
        basicTiling_.baseM = GmmConstants::BASIC_BLOCK_SIZE_128;
    } else if (IsA16MxHiF4(inputParams_)) {
        basicTiling_.baseM =
            inputParams_.hasBias ? GmmConstants::BASIC_BLOCK_SIZE_112 : GmmConstants::BASIC_BLOCK_SIZE_128;
    }

    basicTiling_.baseN = std::min(inputParams_.nSize, GmmConstants::BASIC_BLOCK_SIZE_256);
    basicTiling_.baseN =
        inputParams_.transWeight ?
            CeilAlign(basicTiling_.baseN, GmmConstants::CUBE_BLOCK) :
            CeilAlign(basicTiling_.baseN, GetShapeWithDataType(GmmConstants::L1_ALIGN_SIZE, inputParams_.weightDtype));

    const uint64_t kAlignment = inputParams_.enableMxShapeGeneralization && IsMxA8W4(inputParams_) ?
                                    static_cast<uint64_t>(GmmConstants::MX_GROUP_SIZE_64) :
                                    GetShapeWithDataType(GmmConstants::CUBE_REDUCE_BLOCK, inputParams_.xDtype);
    basicTiling_.baseK = CeilAlign(std::min(GmmConstants::BASIC_BLOCK_SIZE_128, inputParams_.kSize), kAlignment);
}

bool GroupedWeightQuantMatmulAswTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(inputParams_.opName, "workspaces is nullptr."), return false);
    workspaces[0] = GroupedMatmul::SYS_WORKSPACE_SIZE;
    return true;
}

bool GroupedWeightQuantMatmulAswTiling::PostTiling()
{
    context_->SetBlockDim(aicoreParams_.aicNum);
    OP_CHECK_IF(sizeof(tilingData_) % sizeof(uint64_t) != 0,
                OP_LOGE(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8", sizeof(tilingData_)),
                return false);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(inputParams_.opName, "RawTilingData is nullptr."),
                return false);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(inputParams_.opName, "memcpy_s failed, ret = %d", ret);
        return false;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    return true;
}

bool GroupedWeightQuantMatmulAswTiling::SetTilingKey()
{
    tilingKeyConfig_.algorithmCategory = 1;
    tilingKeyConfig_.algorithmSubCategory = 1;
    tilingKeyConfig_.templateCustom = 0;
    tilingKeyConfig_.transA = static_cast<uint8_t>(inputParams_.transX);
    tilingKeyConfig_.transB = static_cast<uint8_t>(inputParams_.transWeight);
    tilingKeyConfig_.antiquantType = static_cast<uint8_t>(inputParams_.weightQuantMode);
    tilingKeyConfig_.quantType = 0;
    tilingKeyConfig_.optionInputSituation = static_cast<uint8_t>(inputParams_.hasAntiquantOffset) << 1;
    tilingKeyConfig_.weightFormat = static_cast<uint8_t>(inputParams_.isWeightNz);

    uint64_t tilingKey = tilingKeyConfig_.GenTilingKey();
    OP_LOGI(inputParams_.opName, "TilingKey: %llu", tilingKey);

    ge::graphStatus ret = context_->SetTilingKey(tilingKey);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "SetTilingKey failed."), return false);
    return true;
}

void GroupedWeightQuantMatmulAswTiling::PrintTilingData()
{
    GroupedMatmulTilingData::GMMWeightQuantAswParams& params = tilingData_.gmmWeightQuantAswParams;
    std::ostringstream oss;
    oss << "GMMWeightQuantAswParams: {groupNum: " << params.groupNum << ", l0cSize: " << params.l0cSize
        << ", singleX: " << static_cast<int32_t>(params.singleX)
        << ", singleW: " << static_cast<int32_t>(params.singleW)
        << ", singleY: " << static_cast<int32_t>(params.singleY)
        << ", groupType: " << static_cast<int32_t>(params.groupType)
        << ", groupListType: " << static_cast<uint32_t>(params.groupListType)
        << ", hasBias: " << static_cast<int32_t>(params.hasBias)
        << ", groupSize: " << static_cast<int32_t>(params.groupSize)
        << "}, TCubeTiling : {M: " << tilingData_.mmTilingData.M << ", N: " << tilingData_.mmTilingData.N
        << ", Ka: " << tilingData_.mmTilingData.Ka << ", usedCoreNum: " << tilingData_.mmTilingData.usedCoreNum
        << ", baseM: " << tilingData_.mmTilingData.baseM << ", baseN: " << tilingData_.mmTilingData.baseN
        << ", baseK: " << tilingData_.mmTilingData.baseK << ", stepKa: " << tilingData_.mmTilingData.stepKa
        << ", stepKb: " << tilingData_.mmTilingData.stepKb
        << ", isBias: " << static_cast<int32_t>(tilingData_.mmTilingData.isBias) << "}";
    OP_LOGI(inputParams_.opName, "%s", oss.str().c_str());
}

uint64_t AswTilingKeyConfigure::GenTilingKey() const
{
    PrintTilingKeyLog();
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(this->weightFormat), static_cast<uint64_t>(this->optionInputSituation),
        static_cast<uint64_t>(this->quantType), static_cast<uint64_t>(this->antiquantType),
        static_cast<uint64_t>(this->transB), static_cast<uint64_t>(this->transA),
        static_cast<uint64_t>(this->templateCustom), static_cast<uint64_t>(WQGMM_NOT_SINGLE_MULTI_SINGLE),
        static_cast<uint64_t>(this->algorithmSubCategory), static_cast<uint64_t>(this->algorithmCategory));
}

} // namespace optiling
