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
#include <algorithm>
#include "err/ops_err.h"
#include "grouped_quant_basic_api_matmul_tiling.h"
#include "../../grouped_matmul_host_util.h"
#include "../../../op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmul;
using namespace optiling::GmmConstant;
using GMMQuantBasicApiTilingData = GroupedMatmulTilingData::GMMQuantBasicApiTilingData;
namespace optiling {
namespace {
constexpr uint64_t THREE_BUFFER_MIN_LOAD_SIZE = MTE2_MIN_LOAD_SIZE_V120 * 3UL / 4UL; // 48 KiB
}

GroupedQmmBasicApiTiling::GroupedQmmBasicApiTiling(gert::TilingContext *context)
    : GroupedQmmTiling(context)
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

    const uint64_t kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    const uint64_t kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    const uint64_t scaleKL1 = std::min(
        std::max(basicTiling_.scaleFactorA * basicTiling_.stepKa, basicTiling_.scaleFactorB * basicTiling_.stepKb) *
            basicTiling_.baseK,
        inputParams_.kSize);
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

    // 仅 NZ weight 通过调整 tiling 优先使能 3-buffer；ND weight 保持原有 tiling 和使能判断逻辑。
    if (inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ) {
        ModifyDepthForUnalign(leftL1Size, baseASize, baseBSize, baseScaleASize + baseScaleBSize);
        return FinalizeStepAndScale();
    }

    const uint64_t initialDepthA1 = basicTiling_.depthA1;
    const uint64_t initialDepthB1 = basicTiling_.depthB1;

    // 先使用初始 stepK 判断是否可以开启 3-buffer。初始值不满足时，搜索 depthA1 和 depthB1 的缩小组合。
    OP_CHECK_IF(FinalizeStepAndScale() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Finalize initial stepK and scale factor failed"),
                return ge::GRAPH_FAILED);
    if (!CanEnableThreeL1Buffer()) {
        L1SearchResult result;
        if (SearchReducedThreeBufferTiling(initialDepthA1, initialDepthB1, baseASize, baseBSize, result)) {
            ApplyThreeBufferSearchResult(result);
            return ge::GRAPH_SUCCESS;
        }
        basicTiling_.depthA1 = initialDepthA1;
        basicTiling_.depthB1 = initialDepthB1;
        ModifyDepthForUnalign(leftL1Size, baseASize, baseBSize, baseScaleASize + baseScaleBSize);
        return FinalizeStepAndScale();
    }

    basicTiling_.depthA1 = initialDepthA1;
    basicTiling_.depthB1 = initialDepthB1;
    ModifyDepthForUnalign(leftL1Size, baseASize, baseBSize, baseScaleASize + baseScaleBSize);
    const uint64_t maxDepthA1 = std::max(basicTiling_.depthA1, initialDepthA1);
    const uint64_t maxDepthB1 = std::max(basicTiling_.depthB1, initialDepthB1);
    L1SearchResult result;
    SearchExpandedThreeBufferTiling(initialDepthA1, initialDepthB1, maxDepthA1, maxDepthB1, baseASize, baseBSize,
                                    result);
    ApplyThreeBufferSearchResult(result);
    return ge::GRAPH_SUCCESS;
}

bool GroupedQmmBasicApiTiling::SearchReducedThreeBufferTiling(uint64_t initialDepthA1, uint64_t initialDepthB1,
                                                              uint64_t baseASize, uint64_t baseBSize,
                                                              L1SearchResult &result)
{
    if (!IsMicroScaling() || inputParams_.transA) {
        return false;
    }

    uint64_t candidateDepthA1 = initialDepthA1;
    while (true) {
        uint64_t candidateDepthB1 = initialDepthB1;
        while (true) {
            TryUpdateThreeBufferCandidate(candidateDepthA1, candidateDepthB1, baseASize, baseBSize, true, result);
            if (candidateDepthB1 <= DB_SIZE) {
                break;
            }
            candidateDepthB1 = std::max(candidateDepthB1 / POWER_OF_TWO, static_cast<uint64_t>(DB_SIZE));
        }

        if (candidateDepthA1 <= DB_SIZE) {
            break;
        }
        candidateDepthA1 = std::max(candidateDepthA1 / POWER_OF_TWO, static_cast<uint64_t>(DB_SIZE));
    }

    return result.found;
}

void GroupedQmmBasicApiTiling::SearchExpandedThreeBufferTiling(uint64_t initialDepthA1, uint64_t initialDepthB1,
                                                               uint64_t maxDepthA1, uint64_t maxDepthB1,
                                                               uint64_t baseASize, uint64_t baseBSize,
                                                               L1SearchResult &result)
{
    uint64_t candidateDepthA1 = initialDepthA1;
    while (true) {
        uint64_t candidateDepthB1 = initialDepthB1;
        while (true) {
            TryUpdateThreeBufferCandidate(candidateDepthA1, candidateDepthB1, baseASize, baseBSize, false, result);
            if (candidateDepthB1 >= maxDepthB1) {
                break;
            }
            candidateDepthB1 = std::min(candidateDepthB1 * POWER_OF_TWO, maxDepthB1);
        }

        if (candidateDepthA1 >= maxDepthA1) {
            break;
        }
        candidateDepthA1 = std::min(candidateDepthA1 * POWER_OF_TWO, maxDepthA1);
    }
}

void GroupedQmmBasicApiTiling::TryUpdateThreeBufferCandidate(uint64_t candidateDepthA1, uint64_t candidateDepthB1,
                                                             uint64_t baseASize, uint64_t baseBSize,
                                                             bool requireMinLoadSize, L1SearchResult &result)
{
    basicTiling_.depthA1 = candidateDepthA1;
    basicTiling_.depthB1 = candidateDepthB1;
    if (FinalizeStepAndScale() != ge::GRAPH_SUCCESS) {
        return;
    }
    if (!CanEnableThreeL1Buffer()) {
        return;
    }

    const uint64_t aPayload = basicTiling_.stepKa * baseASize;
    const uint64_t bPayload = basicTiling_.stepKb * baseBSize;
    // 缩小 depth 以开启 3-buffer 时，避免过小的单次 GM->L1 搬运导致带宽下降。
    if (requireMinLoadSize && (aPayload < THREE_BUFFER_MIN_LOAD_SIZE || bPayload < THREE_BUFFER_MIN_LOAD_SIZE)) {
        return;
    }
    const uint64_t highBwPayload = std::min(aPayload, static_cast<uint64_t>(MTE2_MIN_LOAD_SIZE_V120)) +
                                   std::min(bPayload, static_cast<uint64_t>(MTE2_MIN_LOAD_SIZE_V120));
    const uint64_t totalPayload = aPayload + bPayload;
    if (!IsBetterThreeBufferCandidate(highBwPayload, totalPayload, result)) {
        return;
    }

    result.depthA1 = basicTiling_.depthA1;
    result.depthB1 = basicTiling_.depthB1;
    result.stepKa = basicTiling_.stepKa;
    result.stepKb = basicTiling_.stepKb;
    result.scaleFactorA = basicTiling_.scaleFactorA;
    result.scaleFactorB = basicTiling_.scaleFactorB;
    result.highBwPayload = highBwPayload;
    result.totalPayload = totalPayload;
    result.found = true;
}

bool GroupedQmmBasicApiTiling::IsBetterThreeBufferCandidate(uint64_t highBwPayload, uint64_t totalPayload,
                                                            const L1SearchResult &result) const
{
    return !result.found || highBwPayload > result.highBwPayload ||
           (highBwPayload == result.highBwPayload && totalPayload > result.totalPayload);
}

void GroupedQmmBasicApiTiling::ApplyThreeBufferSearchResult(const L1SearchResult &result)
{
    basicTiling_.depthA1 = result.depthA1;
    basicTiling_.depthB1 = result.depthB1;
    basicTiling_.stepKa = result.stepKa;
    basicTiling_.stepKb = result.stepKb;
    basicTiling_.scaleFactorA = result.scaleFactorA;
    basicTiling_.scaleFactorB = result.scaleFactorB;
}

ge::graphStatus GroupedQmmBasicApiTiling::FinalizeStepAndScale()
{
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

// S4S4 for ARCH35
using namespace GroupedMatmul;

namespace {
constexpr uint32_t DB_SIZE = 2U;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t INT8_NZ_C0 = 32UL;
constexpr uint32_t DATA_SIZE_L0C = 4U;
constexpr uint64_t GMM_MAX_GROUP_LIST_SIZE = 1024UL;
constexpr size_t LAST_FIRST_DIM_INDEX = 1U;
constexpr size_t LAST_SECOND_DIM_INDEX = 2U;
constexpr float EFFECTIVE_TASK_RATIO = 0.95f;

inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return (a + b - 1UL) / b;
}
inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    return CeilDiv(a, b) * b;
}
inline uint64_t GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype)
{
    return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}
// AlignUp in grouped_matmul_tiling.cpp is `static inline` (internal linkage), invisible here; mirror it locally.
template <typename T>
static inline auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}
} // namespace

GroupedS4S4IntQuantTiling::GroupedS4S4IntQuantTiling(gert::TilingContext *context)
    : GroupedQmmBasicApiTiling(context)
{}

void GroupedS4S4IntQuantTiling::Reset()
{
    s4s4Tiling_ = S4S4BasicTiling{};
    int8WeightWs_ = 0UL;
    int8XWs_ = 0UL;
    mmOutWs_ = 0UL;
    perTokenScaleFillWs_ = 0UL;
    weightNzC032_ = false;
}

bool GroupedS4S4IntQuantTiling::IsCapable()
{
    return inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4;
}

ge::graphStatus GroupedS4S4IntQuantTiling::GetPlatformInfo()
{
    OP_CHECK_IF(GroupedQmmBasicApiTiling::GetPlatformInfo() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "S4S4 base GetPlatformInfo failed."),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context_->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
    // AiCoreParams (TilingBaseClass) has no aivNum field; read aivNum from compileInfo directly.
    uint32_t aivNum = compileInfoPtr->aivNum;
    OP_CHECK_IF(aicoreParams_.aicNum == 0 || aivNum != S4S4_AIV_PER_AIC * aicoreParams_.aicNum,
                OP_LOGE(context_->GetNodeName(), "S4S4 mix-core needs aic:aiv=1:2, got aic=%lu aiv=%u",
                        aicoreParams_.aicNum, aivNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(aicoreParams_.ubSize == 0, OP_LOGE(context_->GetNodeName(), "ubSize is 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedS4S4IntQuantTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("GroupedS4S4IntQuantTiling", "context_ is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context_->GetDynamicInputDesc(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wDesc);
    inputParams_.bDtype = wDesc->GetDataType();
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    inputParams_.cDtype = yDesc->GetDataType();
    auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    auto perTokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.aQuantMode = (perTokenScaleDesc != nullptr) ? QuantMode::PERTOKEN_MODE : QuantMode::DEFAULT;
    if (inputParams_.aQuantMode == QuantMode::PERTOKEN_MODE) {
        inputParams_.perTokenScaleDtype = perTokenScaleDesc->GetDataType();
    }
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context_->GetNodeName(), "S4S4: attrs is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_ACT_TYPE + 1,
                OP_LOGE(context_->GetNodeName(), "S4S4: attr num < %lu, got %zu.", ATTR_INDEX_ACT_TYPE + 1,
                        attrs->GetAttrNum()),
                return ge::GRAPH_FAILED);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
    const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    const int64_t *actTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);
    inputParams_.transB = (transposeWeightPtr != nullptr) && *transposeWeightPtr;
    inputParams_.transA = (transposeXPtr != nullptr) && *transposeXPtr;
    inputParams_.groupType = (groupTypePtr != nullptr) ? static_cast<int8_t>(*groupTypePtr) : inputParams_.groupType;
    inputParams_.groupListType =
        (groupListTypePtr != nullptr) ? static_cast<uint8_t>(*groupListTypePtr) : inputParams_.groupListType;
    int64_t actTypeVal = (actTypePtr != nullptr) ? *actTypePtr : 0L;
    OP_CHECK_IF(actTypeVal != 0L, OP_LOGE(context_->GetNodeName(), "S4S4: actType must be 0, got %ld.", actTypeVal),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputParams_.groupType != GroupedMatmul::SPLIT_M,
                OP_LOGE(context_->GetNodeName(), "S4S4: groupType must be 0(SPLIT_M), got %d.",
                        static_cast<int32_t>(inputParams_.groupType)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputParams_.transA, OP_LOGE(context_->GetNodeName(), "S4S4: transA not supported."),
                return ge::GRAPH_FAILED);
    inputParams_.isSingleX = (context_->GetDynamicInputDesc(X_INDEX, 1) == nullptr);
    inputParams_.isSingleW = (context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) == nullptr);
    inputParams_.isSingleY = (context_->GetOutputDesc(1) == nullptr);
    auto xShapePtr = context_->GetDynamicInputShape(X_INDEX, 0);
    auto wShapePtr = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShapePtr);
    const auto &weightStorageShape = wShapePtr->GetStorageShape();
    weightNzC032_ = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && weightStorageShape.GetDimNum() > 0U &&
                    weightStorageShape.GetDim(weightStorageShape.GetDimNum() - 1U) == 32;
    OP_CHECK_IF(!SetMKN(xShapePtr->GetOriginShape(), wShapePtr->GetOriginShape()),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "S4S4 SetMKN failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "S4S4 SetGroupNum failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AnalyzeS4S4(), OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "S4S4 AnalyzeS4S4 failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckS4S4Params(),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "S4S4 CheckS4S4Params failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool GroupedS4S4IntQuantTiling::SetMKN(const gert::Shape &xShape, const gert::Shape &wShape)
{
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    OP_CHECK_IF(xDimNum < MIN_ND_DIM || wDimNum < MIN_ND_DIM,
                OP_LOGE(context_->GetNodeName(), "S4S4: x/weight dim must be >= 2."), return false);
    OP_CHECK_IF(!GroupedQmmBasicApiTiling::SetMKN(xShape, wShape),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "S4S4 base SetMKN failed."), return false);
    s4s4Tiling_.m = inputParams_.mSize;
    s4s4Tiling_.n = inputParams_.nSize;
    s4s4Tiling_.k = inputParams_.kSize;
    s4s4Tiling_.groupType = inputParams_.groupType;
    s4s4Tiling_.groupListType = inputParams_.groupListType;
    return true;
}

bool GroupedS4S4IntQuantTiling::SetGroupNum(uint32_t groupListIndex)
{
    OP_CHECK_IF(!GroupedQmmBasicApiTiling::SetGroupNum(groupListIndex),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "S4S4 base SetGroupNum failed."), return false);
    s4s4Tiling_.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    return true;
}

bool GroupedS4S4IntQuantTiling::SetMKNList()
{
    // Base's mList_/kList_/nList_ is private (inaccessible to derived).
    // Write to our own shadow arrays (mimic base GroupedQmmTiling::SetMKNList).
    if (inputParams_.groupType == GroupedMatmul::SPLIT_M) {
        mList_[0] = -1; // M 动态
        kList_[0] = static_cast<int32_t>(inputParams_.kSize);
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
    } else {
        mList_[0] = static_cast<int32_t>(inputParams_.mSize);
        kList_[0] = -1;
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
    }
    return true;
}

bool GroupedS4S4IntQuantTiling::AnalyzeS4S4()
{
    auto scaleShapePtr = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    uint32_t scaleDimNum = static_cast<uint32_t>(scaleShapePtr->GetStorageShape().GetDimNum());
    if (scaleDimNum == S4S4_PER_GROUP_SCALE_DIM) {
        inputParams_.bQuantMode = QuantMode::PERGROUP_MODE;
        uint64_t g = static_cast<uint64_t>(scaleShapePtr->GetStorageShape().GetDim(1));
        OP_CHECK_IF(g == 0 || inputParams_.kSize % g != 0,
                    OP_LOGE(context_->GetNodeName(), "S4S4 pergroup: k(%lu) must be divisible by G(%lu).",
                            inputParams_.kSize, g),
                    return false);
        OP_CHECK_IF((inputParams_.kSize / g) % S4S4_QUANT_GROUP_SIZE != 0,
                    OP_LOGE(context_->GetNodeName(), "S4S4 pergroup: k/G(%lu) must be even (multiple of 2).",
                            inputParams_.kSize / g),
                    return false);
        s4s4Tiling_.quantGroupNum = static_cast<uint32_t>(g);
    } else if (scaleDimNum == S4S4_PER_CHANNEL_SCALE_DIM) {
        inputParams_.bQuantMode = QuantMode::PERCHANNEL_MODE;
        s4s4Tiling_.quantGroupNum = 1U;
    } else {
        OP_LOGE(context_->GetNodeName(), "S4S4: scale dim should be 2 or 3, got %u.", scaleDimNum);
        return false;
    }
    s4s4Tiling_.isPerTokenQuant = (inputParams_.aQuantMode == QuantMode::PERTOKEN_MODE) ? 1U : 0U;
    return true;
}

bool GroupedS4S4IntQuantTiling::CheckS4S4Params()
{
    OP_CHECK_IF(inputParams_.aDtype != ge::DT_INT4 || inputParams_.bDtype != ge::DT_INT4,
                OP_LOGE(context_->GetNodeName(), "S4S4: x/weight must be INT4."), return false);
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_UINT64,
                OP_LOGE(context_->GetNodeName(), "S4S4: scale must be UINT64, got %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                return false);
    OP_CHECK_IF(inputParams_.aQuantMode == QuantMode::PERTOKEN_MODE && inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "S4S4: perTokenScale must be FLOAT."), return false);
    OP_CHECK_IF(inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "S4S4: y must be FP16/BF16."), return false);
    OP_CHECK_IF(inputParams_.nSize % S4S4_N_ALIGN != 0,
                OP_LOGE(context_->GetNodeName(), "S4S4: n(%lu) must be multiple of 8.", inputParams_.nSize),
                return false);
    OP_CHECK_IF(!inputParams_.isSingleX, OP_LOGE(context_->GetNodeName(), "S4S4 only supports single x."),
                return false);
    OP_CHECK_IF(!inputParams_.isSingleW, OP_LOGE(context_->GetNodeName(), "S4S4 only supports single weight."),
                return false);
    OP_CHECK_IF(!inputParams_.isSingleY, OP_LOGE(context_->GetNodeName(), "S4S4 only supports single y."),
                return false);
    return true;
}

uint32_t GroupedS4S4IntQuantTiling::FindBestSingleN() const
{
    if (s4s4Tiling_.quantGroupNum != 1U || inputParams_.groupType != GroupedMatmul::SPLIT_M) {
        return s4s4Tiling_.baseN;
    }
    if (!inputParams_.isSingleX || !inputParams_.isSingleW) {
        return s4s4Tiling_.baseN;
    }
    uint64_t totalM = s4s4Tiling_.m;
    if (totalM < static_cast<uint64_t>(s4s4Tiling_.baseM)) {
        return s4s4Tiling_.baseN;
    }
    int32_t mDim = static_cast<int32_t>(CeilDiv(totalM, static_cast<uint64_t>(s4s4Tiling_.baseM)));
    int32_t nDim = static_cast<int32_t>(CeilDiv(inputParams_.nSize, static_cast<uint64_t>(s4s4Tiling_.baseN)));
    int64_t taskNum = static_cast<int64_t>(mDim) * nDim * static_cast<int64_t>(s4s4Tiling_.groupNum);
    int64_t usedCore = static_cast<int64_t>(aicoreParams_.aicNum);
    int64_t taskNumPerCore = CeilDiv(taskNum, usedCore);
    if (taskNumPerCore <= 1) {
        return s4s4Tiling_.baseN;
    }
    for (uint32_t i = 1U; i <= static_cast<uint32_t>(usedCore); ++i) {
        uint32_t candidate = static_cast<uint32_t>(CeilDiv(inputParams_.nSize, static_cast<uint64_t>(i)));
        if (candidate % s4s4Tiling_.baseN != 0) {
            continue;
        }
        int32_t curNDim = static_cast<int32_t>(CeilDiv(inputParams_.nSize, static_cast<uint64_t>(candidate)));
        int64_t curTaskNum = static_cast<int64_t>(mDim) * curNDim * static_cast<int64_t>(s4s4Tiling_.groupNum);
        float ratio = static_cast<float>(curTaskNum) / static_cast<float>(AlignUp(curTaskNum, usedCore));
        if (ratio >= EFFECTIVE_TASK_RATIO) {
            return candidate;
        }
    }
    return s4s4Tiling_.baseN;
}

ge::graphStatus GroupedS4S4IntQuantTiling::DoOpTiling()
{
    CalBasicBlockS4S4();
    OP_CHECK_IF(CalUbDivideS4S4() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "CalUbDivideS4S4 failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SetMKNList(), OP_LOGE(context_->GetNodeName(), "SetMKNList failed."), return ge::GRAPH_FAILED);
    s4s4Tiling_.singleN = FindBestSingleN();
    return ge::GRAPH_SUCCESS;
}

void GroupedS4S4IntQuantTiling::CalBasicBlockS4S4()
{
    // The fixed tile fits the Atlas A5 L0 budgets. The kernel masks M/N tails, including M < baseM.
    s4s4Tiling_.baseM = S4S4_BASE_M_950;
    s4s4Tiling_.baseN = S4S4_BASE_N_950;
    if (inputParams_.bQuantMode == QuantMode::PERGROUP_MODE) {
        s4s4Tiling_.baseK = S4S4_BASE_K_950;
    } else {
        s4s4Tiling_.baseK = std::min(static_cast<uint32_t>(inputParams_.kSize), S4S4_BASE_K_950);
    }
    s4s4Tiling_.usedCoreNum = static_cast<uint32_t>(aicoreParams_.aicNum);
}

ge::graphStatus GroupedS4S4IntQuantTiling::CalUbDivideS4S4()
{
    uint32_t ubSize = static_cast<uint32_t>(aicoreParams_.ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context_->GetNodeName(), "ubSize is 0."), return ge::GRAPH_FAILED);

    uint32_t alignBaseN = AlignUp(s4s4Tiling_.baseN, S4S4_N_ALIGN);
    uint32_t ubCalSize = S4S4_VEC_BASE_M_950 * alignBaseN;
    s4s4Tiling_.ubCalSize = ubCalSize;

    uint32_t epilogueBytes = S4S4_EPILOGUE_UB_COEFF * ubCalSize + S4S4_PER_TOKEN_QUEUE_BYTES_PER_M * s4s4Tiling_.baseM;
    OP_CHECK_IF(epilogueBytes >= ubSize,
                OP_LOGE(context_->GetNodeName(), "S4S4 epilogue UB overflow: %u >= %u.", epilogueBytes, ubSize),
                return ge::GRAPH_FAILED);

    // ubRestBytes 给 prologue（32B 对齐）
    uint32_t ubRestBytes = ubSize - epilogueBytes;                       // 154624
    ubRestBytes = ubRestBytes / UB_BLOCK_UNIT_SIZE * UB_BLOCK_UNIT_SIZE; // 154592（32B 对齐）
    OP_CHECK_IF(ubRestBytes < PROLOGUE_MIN_BYTES_950,
                OP_LOGE(context_->GetNodeName(),
                        "S4S4 UB not enough for prologue: ubRestBytes=%u < %u. Consider smaller baseN.", ubRestBytes,
                        PROLOGUE_MIN_BYTES_950),
                return ge::GRAPH_FAILED);
    s4s4Tiling_.ubRestBytes = ubRestBytes;
    return ge::GRAPH_SUCCESS;
}

void GroupedS4S4IntQuantTiling::InitCommonL1TilingFields()
{
    s4s4Tiling_.stepM = 1UL;
    s4s4Tiling_.stepN = 1UL;
    s4s4Tiling_.singleCoreM = std::min(inputParams_.mSize, static_cast<uint64_t>(s4s4Tiling_.baseM));
    s4s4Tiling_.singleCoreN = std::min(inputParams_.nSize, static_cast<uint64_t>(s4s4Tiling_.baseN));
    s4s4Tiling_.singleCoreK = inputParams_.kSize;
    s4s4Tiling_.iterateOrder = 0U;
    s4s4Tiling_.dbL0c = (static_cast<uint64_t>(s4s4Tiling_.baseM) * s4s4Tiling_.baseN * DATA_SIZE_L0C * DB_SIZE <=
                         aicoreParams_.l0cSize) ?
                            DB_SIZE :
                            1U;
}

ge::graphStatus GroupedS4S4IntQuantTiling::CalcLeftL1Size(uint64_t &leftL1Size) const
{
    leftL1Size = aicoreParams_.l1Size;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedS4S4IntQuantTiling::CalL1Tiling()
{
    InitCommonL1TilingFields();
    if (inputParams_.kSize == 0UL) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t leftL1Size = 0UL;
    OP_CHECK_IF(CalcLeftL1Size(leftL1Size) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalcLeftL1Size failed."), return ge::GRAPH_FAILED);
    return CalL1Depth(leftL1Size);
}

ge::graphStatus GroupedS4S4IntQuantTiling::CalL1Depth(uint64_t leftL1Size)
{
    // Validate the double-buffered L0 footprint before deriving the L1 depth.
    OP_CHECK_IF(static_cast<uint64_t>(s4s4Tiling_.baseM) * s4s4Tiling_.baseK * DB_SIZE > aicoreParams_.l0aSize,
                OP_LOGE(context_->GetNodeName(), "S4S4 L0A overflow: need=%lu cap=%lu.",
                        static_cast<uint64_t>(s4s4Tiling_.baseM) * s4s4Tiling_.baseK * DB_SIZE, aicoreParams_.l0aSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(s4s4Tiling_.baseN) * s4s4Tiling_.baseK * DB_SIZE > aicoreParams_.l0bSize,
                OP_LOGE(context_->GetNodeName(), "S4S4 L0B overflow: need=%lu cap=%lu.",
                        static_cast<uint64_t>(s4s4Tiling_.baseN) * s4s4Tiling_.baseK * DB_SIZE, aicoreParams_.l0bSize),
                return ge::GRAPH_FAILED);

    // Each half of L1 stores one aligned A/B/scale stage. kAL1 and kBL1 are the K-element counts of one stage.
    constexpr uint64_t INT8_C0 = 32UL;
    constexpr uint64_t DATA_BLOCK_BYTES = 32UL;
    constexpr uint64_t L1_BUFFER_NUM = DB_SIZE;
    constexpr uint64_t elemBytes = 1UL;
    constexpr uint64_t scaleElemBytes = sizeof(uint64_t);
    auto AlignUpL = [](uint64_t v, uint64_t a) -> uint64_t { return (v + a - 1UL) / a * a; };
    auto GetAL1Bytes = [&](uint64_t kAL1) -> uint64_t {
        return AlignUpL(s4s4Tiling_.baseM, CUBE_BLOCK) * AlignUpL(kAL1, INT8_C0) * elemBytes;
    };
    auto GetBL1Bytes = [&](uint64_t kBL1) -> uint64_t {
        return AlignUpL(kBL1, INT8_C0) * AlignUpL(s4s4Tiling_.baseN, CUBE_BLOCK) * elemBytes;
    };
    auto GetScaleL1Bytes = [&]() -> uint64_t { return AlignUpL(s4s4Tiling_.baseN * scaleElemBytes, DATA_BLOCK_BYTES); };
    auto GetL1StageBytes = [&](uint64_t kAL1, uint64_t kBL1) -> uint64_t {
        return AlignUpL(GetAL1Bytes(kAL1), DATA_BLOCK_BYTES) + AlignUpL(GetBL1Bytes(kBL1), DATA_BLOCK_BYTES) +
               GetScaleL1Bytes();
    };

    const uint64_t halfL1 = leftL1Size / L1_BUFFER_NUM;
    const uint64_t baseAPhys = GetAL1Bytes(s4s4Tiling_.baseK);
    const uint64_t baseBPhys = GetBL1Bytes(s4s4Tiling_.baseK);
    auto floorEven = [](uint64_t v) -> uint64_t {
        return v / static_cast<uint64_t>(DB_SIZE) * static_cast<uint64_t>(DB_SIZE);
    };
    uint64_t depthA1 = floorEven(halfL1 / baseAPhys);
    uint64_t depthB1 = floorEven(halfL1 / baseBPhys);
    while (depthA1 >= DB_SIZE && depthB1 >= DB_SIZE) {
        uint64_t stepKa = depthA1 / DB_SIZE;
        uint64_t stepKb = depthB1 / DB_SIZE;
        if (GetL1StageBytes(stepKa * s4s4Tiling_.baseK, stepKb * s4s4Tiling_.baseK) <= halfL1) {
            break;
        }
        if (baseAPhys * depthA1 >= baseBPhys * depthB1) {
            depthA1 -= DB_SIZE;
        } else {
            depthB1 -= DB_SIZE;
        }
    }
    OP_CHECK_IF(depthA1 < DB_SIZE || depthB1 < DB_SIZE,
                OP_LOGE(context_->GetNodeName(),
                        "S4S4 L1 overflow: cannot fit one fixed-tile stage. baseM=%u baseN=%u baseK=%u halfL1=%lu.",
                        s4s4Tiling_.baseM, s4s4Tiling_.baseN, s4s4Tiling_.baseK, halfL1),
                return ge::GRAPH_FAILED);
    s4s4Tiling_.depthA1 = depthA1;
    s4s4Tiling_.depthB1 = depthB1;
    s4s4Tiling_.stepKa = depthA1 / DB_SIZE;
    s4s4Tiling_.stepKb = depthB1 / DB_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedS4S4IntQuantTiling::DoLibApiTiling()
{
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedS4S4IntQuantTiling::CalWorkspaceS4S4()
{
    // int4->int8 预转换：weight 与 x 均落 workspace（体积翻倍 int4 0.5B -> int8 1B）
    // NZ INT4 uses C0=64, while the expanded INT8 NZ workspace uses C0=32.
    // Therefore its physical [K,N] size must include K/16 and N/32 padding;
    // using logical K*N makes the following X workspace overlap the weight tail.
    const bool weightIsNz = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
    const uint64_t weightK = weightIsNz ? CeilAlign(s4s4Tiling_.k, CUBE_BLOCK) : s4s4Tiling_.k;
    const uint64_t weightN = weightIsNz ? CeilAlign(s4s4Tiling_.n, INT8_NZ_C0) : s4s4Tiling_.n;
    int8WeightWs_ = static_cast<uint64_t>(s4s4Tiling_.groupNum) * weightK * weightN * sizeof(int8_t);
    int8XWs_ = static_cast<uint64_t>(s4s4Tiling_.m) * s4s4Tiling_.k * sizeof(int8_t); // m=totalM (single x)
    mmOutWs_ = static_cast<uint64_t>(S4S4_MMOUT_PIPELINE) * s4s4Tiling_.baseM * s4s4Tiling_.baseN *
               s4s4Tiling_.usedCoreNum * sizeof(uint16_t);
    // F3: perTokenScale 可选兜底。isPerTokenQuant=0 (无 perTokenScale 输入) 时构造全 1 段 (totalM·float),
    //     epilogue 正常 mul (乘 1 等价跳过), 保持 S8S4 epilogue 逐字同步不改。
    perTokenScaleFillWs_ =
        (s4s4Tiling_.isPerTokenQuant == 0U) ? static_cast<uint64_t>(s4s4Tiling_.m) * sizeof(float) : 0UL;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedS4S4IntQuantTiling::GetWorkspaceSize()
{
    OP_CHECK_IF(CalWorkspaceS4S4() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "CalWorkspaceS4S4 failed."),
                return ge::GRAPH_FAILED);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    // 布局: [sys 16MB][int8WeightWs][int8XWs][mmOutWs][perTokenScaleFillWs?]  (?=isPerTokenQuant=0)
    workspaces[0] = SYS_WORKSPACE_SIZE + int8WeightWs_ + int8XWs_ + mmOutWs_ + perTokenScaleFillWs_;
    OP_LOGI(context_->GetNodeName(), "S4S4 workspace: int8Weight=%lu int8X=%lu mmOut=%lu perTokenFill=%lu total=%lu",
            int8WeightWs_, int8XWs_, mmOutWs_, perTokenScaleFillWs_, workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedS4S4IntQuantTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB ? 1 : 0), 0UL, S4S4_KERNEL_TYPE_MIX);
}

ge::graphStatus GroupedS4S4IntQuantTiling::PostTiling()
{
    using namespace GroupedMatmulTilingData;
    auto &p = tilingData_.gmmS4S4Params;
    p.m = static_cast<uint32_t>(s4s4Tiling_.m);
    p.n = static_cast<uint32_t>(s4s4Tiling_.n);
    p.k = static_cast<uint32_t>(s4s4Tiling_.k);
    p.groupNum = s4s4Tiling_.groupNum;
    p.coreNum = s4s4Tiling_.usedCoreNum;
    p.baseM = s4s4Tiling_.baseM;
    p.baseN = s4s4Tiling_.baseN;
    p.baseK = s4s4Tiling_.baseK;
    p.ubCalSize = s4s4Tiling_.ubCalSize;
    p.ubRestBytes = s4s4Tiling_.ubRestBytes;
    p.quantGroupNum = s4s4Tiling_.quantGroupNum;
    p.singleN = s4s4Tiling_.singleN;
    p.isPerTokenQuant = s4s4Tiling_.isPerTokenQuant;
    p.isS4S4Optimize = s4s4Tiling_.isS4S4Optimize;
    p.groupType = s4s4Tiling_.groupType;
    p.groupListType = s4s4Tiling_.groupListType;
    p.reserved = ((inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) ? S4S4_WEIGHT_NZ_FLAG : 0ULL) |
                 (inputParams_.transB ? S4S4_TRANSPOSE_WEIGHT_FLAG : 0ULL) |
                 (weightNzC032_ ? S4S4_WEIGHT_NZ_C0_32_FLAG : 0ULL);
    errno_t retM = memcpy_s(tilingData_.gmmArray.mList, sizeof(tilingData_.gmmArray.mList), mList_, sizeof(mList_));
    OP_CHECK_IF(retM != EOK, OP_LOGE(context_->GetNodeName(), "memcpy_s mList failed, ret=%d", retM),
                return ge::GRAPH_FAILED);
    errno_t retK = memcpy_s(tilingData_.gmmArray.kList, sizeof(tilingData_.gmmArray.kList), kList_, sizeof(kList_));
    OP_CHECK_IF(retK != EOK, OP_LOGE(context_->GetNodeName(), "memcpy_s kList failed, ret=%d", retK),
                return ge::GRAPH_FAILED);
    errno_t retN = memcpy_s(tilingData_.gmmArray.nList, sizeof(tilingData_.gmmArray.nList), nList_, sizeof(nList_));
    OP_CHECK_IF(retN != EOK, OP_LOGE(context_->GetNodeName(), "memcpy_s nList failed, ret=%d", retN),
                return ge::GRAPH_FAILED);
    // mmTilingData(TCubeTiling)
    auto &mm = tilingData_.mmTilingData;
    mm.M = inputParams_.mSize;
    mm.N = inputParams_.nSize;
    mm.Ka = inputParams_.kSize;
    mm.Kb = inputParams_.kSize;
    mm.usedCoreNum = aicoreParams_.aicNum;
    mm.baseM = s4s4Tiling_.baseM;
    mm.baseN = s4s4Tiling_.baseN;
    mm.baseK = s4s4Tiling_.baseK;
    mm.singleCoreM = s4s4Tiling_.singleCoreM;
    mm.singleCoreN = s4s4Tiling_.singleCoreN;
    mm.singleCoreK = s4s4Tiling_.singleCoreK;
    mm.depthA1 = s4s4Tiling_.depthA1;
    mm.depthB1 = s4s4Tiling_.depthB1;
    mm.stepM = s4s4Tiling_.stepM;
    mm.stepN = s4s4Tiling_.stepN;
    mm.stepKa = s4s4Tiling_.stepKa;
    mm.stepKb = s4s4Tiling_.stepKb;
    mm.isBias = 0;
    mm.iterateOrder = s4s4Tiling_.iterateOrder;
    mm.dbL0A = DB_SIZE;
    mm.dbL0B = DB_SIZE;
    mm.dbL0C = s4s4Tiling_.dbL0c;
    OP_LOGI(context_->GetNodeName(),
            "S4S4 tiling: baseM=%u baseN=%u baseK=%u ubCalSize=%u ubRestBytes=%u quantGroupNum=%u isPerToken=%u "
            "depthA1=%lu depthB1=%lu",
            p.baseM, p.baseN, p.baseK, p.ubCalSize, p.ubRestBytes, p.quantGroupNum, p.isPerTokenQuant,
            s4s4Tiling_.depthA1, s4s4Tiling_.depthB1);
    return SaveTilingDataToContext(tilingData_);
}

} // namespace optiling
