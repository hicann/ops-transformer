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
 * \file grouped_quant_basic_api_matmul_tiling.cpp
 * \brief
 */
#include "grouped_quant_basic_api_matmul_tiling.h"
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmul;
using namespace optiling::GmmConstant;
using GMMQuantBasicApiTilingData = GroupedMatmulTilingData::GMMQuantBasicApiTilingData;
namespace optiling {
GroupedQmmBasicApiTiling::GroupedQmmBasicApiTiling(gert::TilingContext *context) : GroupedQmmTiling(context)
{
    Reset();
}

bool GroupedQmmBasicApiTiling::IsCapable()
{
    // MX 量化：scale 为 FLOAT8_E8M0（IsMicroScaling）；支持 mxfp8 与 mxfp4
    if (!IsMicroScaling()) {
        return false;
    }
    const bool isMxFp8 = inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2;
    const bool isMxFp4 = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
    return isMxFp8 || isMxFp4;
}

void GroupedQmmBasicApiTiling::Reset()
{
    tilingData_ = GMMQuantBasicApiTilingData();
}

ge::graphStatus GroupedQmmBasicApiTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

ge::graphStatus GroupedQmmBasicApiTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.activeType = inputParams_.actType;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmQuantParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmQuantParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmQuantParams.groupType = static_cast<int8_t>(inputParams_.groupType);
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmQuantParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    OP_LOGD(inputParams_.opName, "%ld", LogQuantParams(tilingData_.gmmQuantParams));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQmmBasicApiTiling::DoLibApiTiling()
{
    GroupedQmmTiling::CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.m = inputParams_.mSize;
    tilingData_.mmTilingData.n = inputParams_.nSize;
    tilingData_.mmTilingData.k = inputParams_.kSize;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    tilingData_.mmTilingData.kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    tilingData_.mmTilingData.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        tilingData_.mmTilingData.scaleKAL1 = std::min(
            std::max(basicTiling_.scaleFactorA * basicTiling_.stepKa, basicTiling_.scaleFactorB * basicTiling_.stepKb) *
                basicTiling_.baseK,
            inputParams_.kSize);
        tilingData_.mmTilingData.scaleKBL1 = tilingData_.mmTilingData.scaleKAL1;
    }
    tilingData_.mmTilingData.l1BufferStage = static_cast<uint8_t>(CanEnableThreeL1Buffer() ? TB_SIZE : DB_SIZE);
    return ge::GRAPH_SUCCESS;
}

bool GroupedQmmBasicApiTiling::CanEnableThreeL1Buffer() const
{
    if (!IsMicroScaling() || inputParams_.transA) {
        return false;
    }

    const uint64_t kAL1 = tilingData_.mmTilingData.kAL1;
    const uint64_t kBL1 = tilingData_.mmTilingData.kBL1;
    const uint64_t scaleKL1 = tilingData_.mmTilingData.scaleKAL1;
    if (kAL1 == 0UL || kBL1 == 0UL || scaleKL1 == 0UL) {
        return false;
    }

    const uint64_t aL1Size =
        GetSizeWithDataType(basicTiling_.baseM * CeilAlign(kAL1, MX_GROUP_SIZE), inputParams_.aDtype);
    const uint64_t bL1Size =
        GetSizeWithDataType(basicTiling_.baseN * CeilAlign(kBL1, MX_GROUP_SIZE), inputParams_.bDtype);
    const uint64_t scaleKSize = CeilDiv(scaleKL1, MXFP_BASEK_FACTOR) * MXFP_MULTI_BASE_SIZE;
    const uint64_t scaleAL1Size = GetSizeWithDataType(basicTiling_.baseM * scaleKSize, inputParams_.perTokenScaleDtype);
    const uint64_t scaleBL1Size = GetSizeWithDataType(basicTiling_.baseN * scaleKSize, inputParams_.scaleDtype);
    const uint64_t biasL1Size =
        inputParams_.hasBias ? basicTiling_.baseN * ge::GetSizeByDataType(inputParams_.biasDtype) : 0UL;

    const uint64_t commonL1Size = aL1Size + bL1Size + scaleAL1Size + scaleBL1Size + biasL1Size;
    const uint64_t halfL1Size = aicoreParams_.l1Size / NUM_HALF;
    return commonL1Size + aL1Size <= halfL1Size && commonL1Size + bL1Size <= halfL1Size;
}

ge::graphStatus GroupedQmmBasicApiTiling::PostTiling()
{
    return SaveTilingDataToContext(tilingData_);
}

ge::graphStatus GroupedQmmBasicApiTiling::CalL1Tiling()
{
    InitCommonL1TilingFields();
    uint64_t leftL1Size = 0;
    OP_CHECK_IF(CalcLeftL1Size(leftL1Size) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalcLeftL1Size failed"), return ge::GRAPH_FAILED);
    return CalL1Depth(leftL1Size);
}

ge::graphStatus GroupedQmmBasicApiTiling::CalL1Depth(uint64_t leftL1Size)
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

    uint64_t baseScaleASize = 0;
    uint64_t baseScaleBSize = 0;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        CalcAlignedMxBaseScaleSize(baseScaleASize, baseScaleBSize);
    }
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
    OP_CHECK_IF(leftL1Size < baseL1Size,
                OP_LOGE(context_->GetNodeName(), "L1 space overflow. Free L1Size : %lu, used space: %lu", leftL1Size,
                        baseL1Size),
                return ge::GRAPH_FAILED);
    uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL); // 求A+B和的平均depth
    // 根据一条指令带宽要求的数据量求取A,B各自的depth
    basicTiling_.depthA1 = GetDepthWithHighBW(std::min(inputParams_.mSize, basicTiling_.baseM));
    basicTiling_.depthB1 = GetDepthWithHighBW(std::min(inputParams_.nSize, basicTiling_.baseN));
    // 如果按照满足带宽的L1数据量超过了L1Size，进行下调整到平均depth;适配mx低阶api scaleKAL1=scaleKBL1的约束
    if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
            std::max(basicTiling_.depthA1, basicTiling_.depthB1) * (baseScaleASize + baseScaleBSize) >
        leftL1Size) {
        basicTiling_.depthA1 = depthInit;
        basicTiling_.depthB1 = depthInit;
    }
    // 用剩余L1空间对内轴ND非对齐场景进行调整depth
    ModifyDepthForUnalign(leftL1Size, baseASize, baseBSize, baseScaleASize + baseScaleBSize);
    CalStepKs();
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        return CalScaleFactors();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedQmmBasicApiTiling::GetDepthWithHighBW(uint64_t mnL1) const
{
    // 只需要满足读GM数据大于64KB即可获得较高的带宽，不一定要把L1用满，同时减少MTE2头开销
    uint64_t baseKSize = GetSizeWithDataType(basicTiling_.baseK, inputParams_.aDtype);
    uint64_t depth =
        CeilAlign(CeilDiv(MTE2_MIN_LOAD_SIZE_V120, mnL1), static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256)) /
        baseKSize * DB_SIZE;
    uint64_t pow2Depth = POWER_OF_TWO;
    while (pow2Depth < depth) {
        pow2Depth *= POWER_OF_TWO;
    }
    // 对齐2次幂或者实际最大depth大小
    return std::min(pow2Depth, CeilDiv(inputParams_.kSize, basicTiling_.baseK) * DB_SIZE);
}

void GroupedQmmBasicApiTiling::ModifyDepthForUnalign(uint64_t leftL1Size, uint64_t baseASize, uint64_t baseBSize,
                                                     uint64_t baseScaleABSize)
{
    // 只调整K轴非对齐场景
    if (inputParams_.kSize % GmmConstant::BASIC_BLOCK_SIZE_128 == 0) {
        return;
    }
    // m，n在内轴且ND时，修改stepk无法改变ND2NZ小包数量
    if (inputParams_.transA && (!inputParams_.transB || inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ)) {
        return;
    }
    if (!inputParams_.transA) {
        if (basicTiling_.depthA1 <= basicTiling_.depthB1) {
            uint64_t leftASize = leftL1Size - basicTiling_.depthB1 * baseBSize - basicTiling_.depthB1 * baseScaleABSize;
            while (basicTiling_.depthA1 * POWER_OF_TWO * baseASize <= leftASize) {
                basicTiling_.depthA1 *= POWER_OF_TWO;
            }
            if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
                    std::max(basicTiling_.depthA1, basicTiling_.depthB1) * baseScaleABSize >
                leftL1Size) {
                basicTiling_.depthA1 = basicTiling_.depthB1;
            }
        } else if (inputParams_.transB && inputParams_.bFormat == ge::FORMAT_ND) {
            uint64_t leftBSize = leftL1Size - basicTiling_.depthA1 * baseASize - basicTiling_.depthA1 * baseScaleABSize;
            while (basicTiling_.depthB1 * POWER_OF_TWO * baseBSize <= leftBSize) {
                basicTiling_.depthB1 *= POWER_OF_TWO;
            }
            if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
                    std::max(basicTiling_.depthA1, basicTiling_.depthB1) * baseScaleABSize >
                leftL1Size) {
                basicTiling_.depthB1 = basicTiling_.depthA1;
            }
        }
    } else { // transA = true, transB = true, 仅考虑B depth
        while ((basicTiling_.depthA1 * baseASize -
                std::max(basicTiling_.depthA1, basicTiling_.depthB1 * POWER_OF_TWO) * baseScaleABSize) < leftL1Size) {
            basicTiling_.depthB1 *= POWER_OF_TWO;
        }
    }
}

ge::graphStatus GroupedQmmBasicApiTiling::CalScaleFactors()
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);
    uint64_t baseScaleASize = GetSizeWithDataType(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseM,
                                                  inputParams_.perTokenScaleDtype);
    uint64_t baseScaleBSize =
        GetSizeWithDataType(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseN, inputParams_.scaleDtype);
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t leftL1Size =
        aicoreParams_.l1Size - (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(
        leftL1Size / (std::max(basicTiling_.depthA1, basicTiling_.depthB1) * (baseScaleASize + baseScaleBSize)));
    OP_CHECK_IF(
        scaleInit == 0,
        OP_LOGE(context_->GetNodeName(),
                "When m(%lu)/n(%lu)/k(%lu)/groupNum(%lu) in mx quant mode, scaleFactor should not be equal to 0.",
                inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum),
        return ge::GRAPH_FAILED);
    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleASize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleBSize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA =
        static_cast<uint32_t>(CeilDiv(inputParams_.kSize, basicTiling_.stepKa * basicTiling_.baseK));
    uint32_t scaleFactorB =
        static_cast<uint32_t>(CeilDiv(inputParams_.kSize, basicTiling_.stepKb * basicTiling_.baseK));
    basicTiling_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    basicTiling_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    basicTiling_.scaleFactorA = std::min(scaleFactorAMax, basicTiling_.scaleFactorA);
    basicTiling_.scaleFactorB = std::min(scaleFactorBMax, basicTiling_.scaleFactorB);

    // 来自L1 size 的约束
    if (basicTiling_.scaleFactorA > scaleInit && basicTiling_.scaleFactorB > scaleInit) { // 非scalek全载，ka/kb倍数
        if (basicTiling_.depthA1 >= basicTiling_.depthB1) {
            basicTiling_.scaleFactorA = scaleInit;
            basicTiling_.scaleFactorB = scaleInit * basicTiling_.depthA1 / basicTiling_.depthB1;
        } else {
            basicTiling_.scaleFactorA = scaleInit * basicTiling_.depthB1 / basicTiling_.depthA1;
            basicTiling_.scaleFactorB = scaleInit;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
