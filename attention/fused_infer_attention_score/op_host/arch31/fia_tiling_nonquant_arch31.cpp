/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fia_tiling_nonquant_arch31.cpp
 * \brief arch31 (KirinX90 / Kirin9030) non-quant Tiling 实现
 */

#include "fia_tiling_nonquant_arch31.h"
#include "../fused_infer_attention_score_tiling.h"
#include "../fia_tiling_info.h"
#include "fused_infer_attention_score_tiling_check_arch31.h"
#include "../fused_infer_attention_score_tiling_index.h"
#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

namespace {
constexpr uint64_t FIA_ARCH31_FLAG_TILING = 6000000000000000000ULL;
constexpr uint32_t ARCH31_CUBE_M = 16;
constexpr uint32_t ARCH31_CUBE_K = 16;
constexpr uint32_t ARCH31_UB_BUDGET_BYTES = 90 * 1024;
constexpr uint32_t ARCH31_M_BASE_D256 = 32;
constexpr uint32_t ARCH31_M_BASE_D512_DAV3003 = 16;
constexpr uint32_t ARCH31_M_BASE_D512_DAV3113_UNSPLIT = 8;
constexpr uint32_t ARCH31_M_BASE_UPGRADED = 16;
constexpr uint32_t ARCH31_M_BASE_DEFAULT = ARCH31_M_BASE_D256;
constexpr uint32_t ARCH31_M_BASE_D512 = ARCH31_M_BASE_D512_DAV3003;
constexpr uint32_t ARCH31_N_BASE_UB = 128;
constexpr uint32_t ARCH31_D_INNER_DEFAULT = 256;
constexpr uint64_t ARCH31_FP16_BYTES = 2;

constexpr uint32_t ARCH31_LAYOUT_BNSD = 0;
constexpr uint32_t ARCH31_LAYOUT_BSND = 1;
constexpr uint32_t ARCH31_LAYOUT_BSH = 2;
constexpr uint32_t ARCH31_LAYOUT_TND = 3;

constexpr uint32_t ARCH31_HP_PATH_BIT_PREFILL = 0;
constexpr uint32_t ARCH31_HP_PATH_BIT_DECODE = 1;
constexpr uint32_t ARCH31_HP_DIM_BIT_256 = 0;
constexpr uint32_t ARCH31_HP_DIM_BIT_512 = 1;
constexpr uint32_t ARCH31_HP_CODE_BIAS = 1;
constexpr uint32_t ARCH31_HP_S1_PREFILL = 64;
constexpr uint32_t ARCH31_HP_S1_DECODE = 1;
constexpr uint32_t ARCH31_HP_DIM_256 = 256;
constexpr uint32_t ARCH31_HP_DIM_512 = 512;

constexpr uint32_t UB_COEFF_O_ACC = 4;
constexpr uint32_t UB_COEFF_MM2RES = 4;
constexpr uint32_t UB_COEFF_OUTFP16 = 2;
constexpr uint32_t UB_COEFF_ACCUM = 12;
constexpr uint32_t UB_COEFF_MASK_PER_NDEAL = 1;

inline uint32_t Arch31UbMUpper(uint32_t dCols, uint32_t nDealMax)
{
    uint64_t scratchCols = std::max<uint64_t>(nDealMax, dCols);
    uint64_t perRow = scratchCols * (UB_COEFF_MM2RES + UB_COEFF_OUTFP16) +
                      static_cast<uint64_t>(dCols) * UB_COEFF_O_ACC + UB_COEFF_ACCUM +
                      static_cast<uint64_t>(nDealMax) * UB_COEFF_MASK_PER_NDEAL;
    if (perRow == 0) {
        return ARCH31_M_BASE_D256;
    }
    uint32_t m = static_cast<uint32_t>(ARCH31_UB_BUDGET_BYTES / perRow);
    return (m / ARCH31_CUBE_M) * ARCH31_CUBE_M;
}

constexpr uint32_t ARCH31_TND_MAX_BATCH = 128;
constexpr uint64_t ARCH31_TND_WS_INT32_MAX = 2147483647ULL;

struct Arch31Dims {
    uint32_t b = 0;
    uint32_t heads = 0;
    uint32_t kvHeads = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t dQ = 0;
    uint32_t dK = 0;
    uint32_t dV = 0;
};

bool ParseLayoutDimsArch31(const gert::Shape& queryShape, const gert::Shape& keyShape, const gert::Shape& valueShape,
                           const std::string& layout, uint32_t headsAttr, uint32_t kvHeadsAttr, Arch31Dims& dims,
                           const int64_t* seqQVals = nullptr, uint32_t seqQNum = 0, const int64_t* seqKvVals = nullptr,
                           uint32_t seqKvNum = 0)
{
    if (layout == "BNSD") {
        if (queryShape.GetDimNum() != 4 || keyShape.GetDimNum() != 4 || valueShape.GetDimNum() != 4) {
            return false;
        }
        dims.b = static_cast<uint32_t>(queryShape.GetDim(0));
        dims.heads = static_cast<uint32_t>(queryShape.GetDim(1));
        dims.s1 = static_cast<uint32_t>(queryShape.GetDim(2));
        dims.dQ = static_cast<uint32_t>(queryShape.GetDim(3));
        dims.kvHeads = static_cast<uint32_t>(keyShape.GetDim(1));
        dims.s2 = static_cast<uint32_t>(keyShape.GetDim(2));
        dims.dK = static_cast<uint32_t>(keyShape.GetDim(3));
        dims.dV = static_cast<uint32_t>(valueShape.GetDim(3));
        return true;
    } else if (layout == "BSND") {
        if (queryShape.GetDimNum() != 4 || keyShape.GetDimNum() != 4 || valueShape.GetDimNum() != 4) {
            return false;
        }
        dims.b = static_cast<uint32_t>(queryShape.GetDim(0));
        dims.s1 = static_cast<uint32_t>(queryShape.GetDim(1));
        dims.heads = static_cast<uint32_t>(queryShape.GetDim(2));
        dims.dQ = static_cast<uint32_t>(queryShape.GetDim(3));
        dims.s2 = static_cast<uint32_t>(keyShape.GetDim(1));
        dims.kvHeads = static_cast<uint32_t>(keyShape.GetDim(2));
        dims.dK = static_cast<uint32_t>(keyShape.GetDim(3));
        dims.dV = static_cast<uint32_t>(valueShape.GetDim(3));
        return true;
    } else if (layout == "BSH") {
        if (queryShape.GetDimNum() != 3 || keyShape.GetDimNum() != 3 || valueShape.GetDimNum() != 3) {
            return false;
        }
        if (headsAttr == 0 || kvHeadsAttr == 0) {
            return false;
        }
        int64_t qFused = queryShape.GetDim(2);
        int64_t kFused = keyShape.GetDim(2);
        int64_t vFused = valueShape.GetDim(2);
        if (qFused % static_cast<int64_t>(headsAttr) != 0 || kFused % static_cast<int64_t>(kvHeadsAttr) != 0 ||
            vFused % static_cast<int64_t>(kvHeadsAttr) != 0) {
            return false;
        }
        dims.b = static_cast<uint32_t>(queryShape.GetDim(0));
        dims.s1 = static_cast<uint32_t>(queryShape.GetDim(1));
        dims.heads = headsAttr;
        dims.dQ = static_cast<uint32_t>(qFused / static_cast<int64_t>(headsAttr));
        dims.s2 = static_cast<uint32_t>(keyShape.GetDim(1));
        dims.kvHeads = kvHeadsAttr;
        dims.dK = static_cast<uint32_t>(kFused / static_cast<int64_t>(kvHeadsAttr));
        dims.dV = static_cast<uint32_t>(vFused / static_cast<int64_t>(kvHeadsAttr));
        return true;
    } else if (layout == "TND") {
        if (queryShape.GetDimNum() != 3 || keyShape.GetDimNum() != 3 || valueShape.GetDimNum() != 3) {
            return false;
        }
        if (seqQVals == nullptr || seqKvVals == nullptr || seqQNum == 0 || seqQNum != seqKvNum ||
            seqQNum > ARCH31_TND_MAX_BATCH) {
            return false;
        }
        dims.b = seqQNum;
        dims.heads = static_cast<uint32_t>(queryShape.GetDim(1));
        dims.dQ = static_cast<uint32_t>(queryShape.GetDim(2));
        dims.kvHeads = static_cast<uint32_t>(keyShape.GetDim(1));
        dims.dK = static_cast<uint32_t>(keyShape.GetDim(2));
        dims.dV = static_cast<uint32_t>(valueShape.GetDim(2));
        uint32_t s1Max = 0;
        uint32_t s2Max = 0;
        for (uint32_t i = 0; i < seqQNum; i++) {
            int64_t qEnd = seqQVals[i];
            int64_t qBegin = (i == 0) ? 0 : seqQVals[i - 1];
            int64_t kvEnd = seqKvVals[i];
            int64_t kvBegin = (i == 0) ? 0 : seqKvVals[i - 1];
            uint32_t s1 = (qEnd > qBegin) ? static_cast<uint32_t>(qEnd - qBegin) : 0;
            uint32_t s2 = (kvEnd > kvBegin) ? static_cast<uint32_t>(kvEnd - kvBegin) : 0;
            s1Max = (s1 > s1Max) ? s1 : s1Max;
            s2Max = (s2 > s2Max) ? s2 : s2Max;
        }
        dims.s1 = s1Max;
        dims.s2 = s2Max;
        return true;
    } else {
        return false;
    }
}

bool LoadTndSeqPtrs(const gert::TilingContext* context, const int64_t*& seqQVals, uint32_t& seqQNum,
                    const int64_t*& seqKvVals, uint32_t& seqKvNum)
{
    seqQVals = nullptr;
    seqKvVals = nullptr;
    seqQNum = 0;
    seqKvNum = 0;
    auto seqQShape = context->GetOptionalInputShape(ACTUAL_SEQ_Q_INDEX);
    auto seqKvShape = context->GetOptionalInputShape(ACTUAL_SEQ_KV_INDEX);
    auto seqQTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto seqKvTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    if (seqQShape == nullptr || seqKvShape == nullptr) {
        return false;
    }
    seqQNum = static_cast<uint32_t>(seqQShape->GetStorageShape().GetDim(0));
    seqKvNum = static_cast<uint32_t>(seqKvShape->GetStorageShape().GetDim(0));
    if (seqQTensor != nullptr) {
        seqQVals = seqQTensor->GetData<int64_t>();
    }
    if (seqKvTensor != nullptr) {
        seqKvVals = seqKvTensor->GetData<int64_t>();
    }
    return seqQVals != nullptr && seqKvVals != nullptr && seqQNum > 0 && seqQNum == seqKvNum &&
           seqQNum <= ARCH31_TND_MAX_BATCH;
}

inline uint64_t Arch31S2StridePad(uint64_t s2)
{
    return (s2 + 15ULL) & ~15ULL;
}

void AccumulateTndSums(const int64_t* seqQVals, uint32_t seqQNum, const int64_t* seqKvVals, uint64_t& sumS1S2,
                       uint64_t& sumS1)
{
    sumS1S2 = 0;
    sumS1 = 0;
    for (uint32_t i = 0; i < seqQNum; i++) {
        int64_t qEnd = seqQVals[i];
        int64_t qBegin = (i == 0) ? 0 : seqQVals[i - 1];
        int64_t kvEnd = seqKvVals[i];
        int64_t kvBegin = (i == 0) ? 0 : seqKvVals[i - 1];
        uint64_t s1 = (qEnd > qBegin) ? static_cast<uint64_t>(qEnd - qBegin) : 0;
        uint64_t s2 = (kvEnd > kvBegin) ? static_cast<uint64_t>(kvEnd - kvBegin) : 0;
        sumS1S2 += s1 * Arch31S2StridePad(s2);
        sumS1 += s1;
    }
}

bool IsDecodeSparseBypass(const gert::TilingContext* context, const Arch31Dims& dims, const std::string& inputLayout)
{
    if (inputLayout != "BNSD" && inputLayout != "BSND" && inputLayout != "BSH") {
        return false;
    }
    if (dims.s1 != 1) {
        return false;
    }
    if (dims.dQ != dims.dK || dims.dQ != dims.dV) {
        return false;
    }
    if (context->GetOptionalInputShape(QUERY_ROPE_INDEX) != nullptr ||
        context->GetOptionalInputShape(KEY_ROPE_INDEX) != nullptr) {
        return false;
    }
    return true;
}
} // namespace

ge::graphStatus FiaTilingNonQuantArch31::DoTiling()
{
    // 1. 平台信息获取
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(), "GetPlatformInfo returned nullptr"}));
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    // 2. 获取输入 shape
    auto queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto valueShape = context_->GetInputShape(2)->GetStorageShape();

    // 3 获取属性
    auto attrs = context_->GetAttrs();
    float scaleValue = 1.0f;
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    std::string inputLayout = "BNSD";
    int64_t sparseMode = 0;
    int64_t softmaxLseFlag = 0;
    uint32_t kvHeadNumsAttr = 0;
    int32_t numHeadsAttr = 0;
    if (attrs != nullptr) {
        auto numHeadsAttrPtr = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
        if (numHeadsAttrPtr != nullptr) {
            numHeadsAttr = static_cast<uint32_t>(*numHeadsAttrPtr);
        }
        auto scaleAttr = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
        if (scaleAttr != nullptr) {
            scaleValue = *scaleAttr;
        }
        auto preTokensAttr = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
        if (preTokensAttr != nullptr) {
            preTokens = *preTokensAttr;
        }
        auto nextTokensAttr = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
        if (nextTokensAttr != nullptr) {
            nextTokens = *nextTokensAttr;
        }
        auto layoutStr = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
        if (layoutStr != nullptr) {
            inputLayout = std::string(layoutStr);
        }
        auto kvHeadNumsAttrPtr = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
        if (kvHeadNumsAttrPtr != nullptr) {
            kvHeadNumsAttr = static_cast<uint32_t>(*kvHeadNumsAttrPtr);
        }
        auto sparseAttr = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
        if (sparseAttr != nullptr) {
            sparseMode = static_cast<int64_t>(*sparseAttr);
        }
        auto lseFlagAttr = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
        if (lseFlagAttr != nullptr) {
            softmaxLseFlag = (*lseFlagAttr) ? 1 : 0;
        }
    }

    const int64_t* seqQVals = nullptr;
    const int64_t* seqKvVals = nullptr;
    uint32_t seqQNum = 0;
    uint32_t seqKvNum = 0;
    if (inputLayout == "TND" && !LoadTndSeqPtrs(context_, seqQVals, seqQNum, seqKvVals, seqKvNum)) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "actual_seq_lengths/actual_seq_lengths_kv");
        return ge::GRAPH_FAILED;
    }

    Arch31Dims dims;
    if (!ParseLayoutDimsArch31(queryShape, keyShape, valueShape, inputLayout, static_cast<uint32_t>(numHeadsAttr),
                               kvHeadNumsAttr, dims, seqQVals, seqQNum, seqKvVals, seqKvNum)) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      (std::string("arch31 layout ") + inputLayout +
                                       " shape parse failed: query dimNum=" + std::to_string(queryShape.GetDimNum()) +
                                       ", key dimNum=" + std::to_string(keyShape.GetDimNum()) +
                                       ", value dimNum=" + std::to_string(valueShape.GetDimNum()))
                                          .c_str()}));
        return ge::GRAPH_FAILED;
    }
    uint32_t B = dims.b;
    uint32_t H = dims.heads;
    uint32_t S1 = dims.s1;
    uint32_t D = dims.dQ;
    uint32_t KvH = dims.kvHeads;
    uint32_t S2 = dims.s2;

    // GQA: group = H / KvH
    uint32_t group = 1;
    if (KvH > 0) {
        group = H / KvH;
    }

    // 4. 计算切D 分档 + M_BASE / N_BASE / D_inner （按 Get Core MemSize + UB 反压联合分档）
    uint32_t mBase = 0;
    uint32_t nBase = 0;
    uint32_t dInner = 0;
    bool needSplitD = false;
    uint32_t dTile = 0;
    uint32_t kTileQK = 0;
    uint32_t nTileV = 0;
    uint32_t mBaseDeal = 0;
    uint32_t kTileV = 0;
    if (CalcTilingParams(mBase, nBase, dInner, needSplitD, dTile, kTileQK, nTileV, mBaseDeal, kTileV) !=
        ge::GRAPH_SUCCESS) {
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context_->GetNodeName(), "CalcTilingParams failed"}));
        return ge::GRAPH_FAILED;
    }

    bool decodeGateHit = IsDecodeSparseBypass(context_, dims, inputLayout);
    bool analyticMaskFlag = (sparseMode == 2 || sparseMode == 3 || sparseMode == 4) && !decodeGateHit;
    auto attenMaskShapeOpt = context_->GetOptionalInputShape(ATTEN_MASK_INDEX);
    bool attenMaskTensorFlag =
        (attenMaskShapeOpt != nullptr) && (sparseMode == 0 || sparseMode == 1 || sparseMode == 9);

    const bool hpPathOk = (S1 == ARCH31_HP_S1_PREFILL || S1 == ARCH31_HP_S1_DECODE);
    const bool hpDimOk = (D == ARCH31_HP_DIM_256 || D == ARCH31_HP_DIM_512);
    const bool hpShapeHit = hpPathOk && hpDimOk && B == 1 && H == 8 && KvH == 2 && (S2 == 2048 || S2 == 576) &&
                            group == 4 && inputLayout == "BNSD" && scaleValue == 1.0f && softmaxLseFlag == 0 &&
                            attenMaskTensorFlag && sparseMode == 0 && !needSplitD;
    const uint32_t pathBit = (S1 == ARCH31_HP_S1_DECODE) ? ARCH31_HP_PATH_BIT_DECODE : ARCH31_HP_PATH_BIT_PREFILL;
    const uint32_t dimBit = (D == ARCH31_HP_DIM_512) ? ARCH31_HP_DIM_BIT_512 : ARCH31_HP_DIM_BIT_256;
    const uint32_t hpCode = hpShapeHit ? ARCH31_HP_CODE_BIAS + (pathBit << 1) + dimBit : 0U;
    const bool isHp = (hpCode != 0U);
    const bool hpPrefill = isHp && (pathBit == ARCH31_HP_PATH_BIT_PREFILL);

    // 5. 计算 TilingKey
    uint32_t layoutCode = ARCH31_LAYOUT_BNSD;
    if (inputLayout == "BSND") {
        layoutCode = ARCH31_LAYOUT_BSND;
    } else if (inputLayout == "BSH") {
        layoutCode = ARCH31_LAYOUT_BSH;
    } else if (inputLayout == "TND") {
        layoutCode = ARCH31_LAYOUT_TND;
    }
    uint64_t tilingKey = CalcTilingKey(static_cast<uint32_t>(sparseMode), layoutCode, attenMaskTensorFlag, hpCode);
    context_->SetTilingKey(tilingKey);
    OP_LOGI(context_->GetNodeName(), "arch31 tilingKey=%llu hpCode=%u (S1=%u D=%u preTokens=%lld)",
            static_cast<unsigned long long>(tilingKey), hpCode, S1, D, static_cast<long long>(preTokens));

    // 6. 计算workspace 大小
    uint64_t workspaceSize = 0;
    if (CalcWorkspaceSize(workspaceSize, mBase, nBase, dInner) != ge::GRAPH_SUCCESS) {
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context_->GetNodeName(), "CalcWorkspaceSize failed"}));
        return ge::GRAPH_FAILED;
    }
    size_t* workSpaces = context_->GetWorkspaceSizes(1);
    if (workSpaces != nullptr) {
        workSpaces[0] = static_cast<size_t>(workspaceSize);
    } else {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(), "GetWorkspaceSizes returned nullptr"}));
        return ge::GRAPH_FAILED;
    }

    // 7. 写入 TilingData（复用既有 FusedInferAttentionScoreTilingData 子集）
    auto* tilingDataPtr = &tilingData_;
    tilingDataPtr->baseParams.set_bSize(B);
    tilingDataPtr->baseParams.set_n2Size(KvH);
    tilingDataPtr->baseParams.set_gSize(group);
    tilingDataPtr->baseParams.set_s1Size(S1);
    tilingDataPtr->baseParams.set_s2Size(S2);
    tilingDataPtr->baseParams.set_headDim(D);
    tilingDataPtr->baseParams.set_headDimRope(D);
    if (inputLayout == "TND") {
        tilingDataPtr->baseParams.set_actualSeqS1Dims(B);
        tilingDataPtr->baseParams.set_actualSeqS2Dims(B);
        tilingDataPtr->baseParams.set_accumQSeqFlag(1);
        tilingDataPtr->baseParams.set_accumKVSeqFlag(1);
    } else {
        tilingDataPtr->baseParams.set_actualSeqS1Dims(0);
        tilingDataPtr->baseParams.set_actualSeqS2Dims(0);
        tilingDataPtr->baseParams.set_accumQSeqFlag(0);
        tilingDataPtr->baseParams.set_accumKVSeqFlag(0);
    }
    tilingDataPtr->baseParams.set_scaleValue(scaleValue);
    tilingDataPtr->baseParams.set_usedCoreNum(1);
    tilingDataPtr->baseParams.set_outputLayout(layoutCode);
    tilingDataPtr->baseParams.set_batchContinuous(1);
    tilingDataPtr->baseParams.set_softmaxLseFlag(static_cast<uint32_t>(softmaxLseFlag));
    tilingDataPtr->baseParams.set_needInit(0);
    tilingDataPtr->baseParams.set_slidingFlag(0);
    tilingDataPtr->baseParams.set_l2CacheOffFlag(1);
    tilingDataPtr->baseParams.set_isLegacyIfa(0);
    tilingDataPtr->baseParams.set_isEmptyBatchOverHalf(false);
    tilingDataPtr->pageAttenParams.set_blockSize(0);
    tilingDataPtr->pageAttenParams.set_maxBlockNumPerBatch(0);
    uint32_t maskFlagField = (analyticMaskFlag ? 1 : 0) | (attenMaskTensorFlag ? 2 : 0);
    tilingDataPtr->maskParams.set_attenMaskFlag(maskFlagField);
    uint32_t attenMaskBatchStride = 0;
    uint32_t attenMaskStride = 0;
    if (attenMaskTensorFlag && attenMaskShapeOpt != nullptr) {
        if (inputLayout == "TND") {
            attenMaskBatchStride = 0;
            attenMaskStride = S1;
        } else if (sparseMode == 9) {
            attenMaskBatchStride = (B > 1) ? (S1 * S1) : 0;
            attenMaskStride = S1;
        } else {
            auto maskShape = attenMaskShapeOpt->GetStorageShape();
            bool batched = B > 1 && maskShape.GetDim(0) == static_cast<int64_t>(B);
            attenMaskBatchStride = batched ? (S1 * S2) : 0;
            attenMaskStride = S2;
        }
    }
    tilingDataPtr->maskParams.set_attenMaskBatchStride(attenMaskBatchStride);
    tilingDataPtr->maskParams.set_attenMaskStride(attenMaskStride);
    tilingDataPtr->maskParams.set_preToken(static_cast<int32_t>(preTokens));
    tilingDataPtr->maskParams.set_nextToken(static_cast<int32_t>(nextTokens));
    tilingDataPtr->maskParams.set_isRowInvalid(0);
    tilingDataPtr->maskParams.set_isExistRowInvalid(0);
    tilingDataPtr->maskParams.set_sparseMode(static_cast<uint32_t>(sparseMode));

    uint32_t nTileVOut = nTileV;
    uint32_t kTileVOut = kTileV;
    uint32_t kTileQKOut = kTileQK;
    uint32_t mBaseDealOut = mBaseDeal;
    uint32_t nBaseOut = nBase;
    if (isHp) {
        nTileVOut = dTile;
        kTileVOut = dTile;
        kTileQKOut = dTile;
        if (kTileQKOut < ARCH31_CUBE_K) {
            kTileQKOut = ARCH31_CUBE_K;
        }
        uint64_t l0bSizeB = 0;
        uint64_t l0cSizeB = 0;
        uint64_t ubSizeB = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSizeB);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSizeB);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeB);
        OP_CHECK_IF(l0bSizeB == 0 || l0cSizeB == 0 || ubSizeB == 0,
                    REPORT_PREDEFINED_ERR_MSG(
                        "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                        (std::vector<const char*>{
                            context_->GetNodeName(),
                            (std::string("GetCoreMemSize returned 0 (l0b=") + std::to_string(l0bSizeB) +
                             ", l0c=" + std::to_string(l0cSizeB) + ", ub=" + std::to_string(ubSizeB) + ")")
                                .c_str()})),
                    return ge::GRAPH_FAILED);
        uint32_t nDealMaxB = static_cast<uint32_t>(l0bSizeB / (static_cast<uint64_t>(dTile) * ARCH31_FP16_BYTES));
        nDealMaxB = (nDealMaxB / ARCH31_CUBE_M) * ARCH31_CUBE_M;
        if (nDealMaxB > ARCH31_N_BASE_UB) {
            nDealMaxB = ARCH31_N_BASE_UB;
        }
        if (nDealMaxB < ARCH31_CUBE_M) {
            nDealMaxB = ARCH31_CUBE_M;
        }
        nBaseOut = nDealMaxB;
        kTileVOut = nDealMaxB;

        uint32_t mDealL0C = static_cast<uint32_t>(l0cSizeB / (static_cast<uint64_t>(dTile) * ARCH31_FP16_BYTES * 2ULL));
        mDealL0C = (mDealL0C / ARCH31_CUBE_M) * ARCH31_CUBE_M;
        uint32_t mDealUB = Arch31UbMUpper(dTile, nDealMaxB);
        uint32_t mDealB = std::min(mDealL0C, mDealUB);
        if (mDealB < ARCH31_CUBE_M) {
            mDealB = ARCH31_CUBE_M;
        }
        uint32_t mNeedB = (S1 < mBaseDeal) ? S1 : mBaseDeal;
        mNeedB = ((mNeedB + ARCH31_CUBE_M - 1) / ARCH31_CUBE_M) * ARCH31_CUBE_M;
        if (mDealB > mNeedB) {
            mDealB = mNeedB;
        }
        mBaseDealOut = mDealB;
        mBase = mDealB;
        nBase = nDealMaxB;

        OP_CHECK_IF(static_cast<uint64_t>(nDealMaxB) * dTile * ARCH31_FP16_BYTES > l0bSizeB,
                    REPORT_PREDEFINED_ERR_MSG(
                        "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                        (std::vector<const char*>{
                            context_->GetNodeName(),
                            (std::string("HP L0B overflow: nDeal=") + std::to_string(nDealMaxB) +
                             ", D=" + std::to_string(dTile) +
                             ", need=" + std::to_string(static_cast<uint64_t>(nDealMaxB) * dTile * ARCH31_FP16_BYTES) +
                             " > l0b=" + std::to_string(l0bSizeB))
                                .c_str()})),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint64_t>(2) * mDealB * dTile * ARCH31_FP16_BYTES * 2ULL > l0cSizeB,
                    REPORT_PREDEFINED_ERR_MSG(
                        "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                        (std::vector<const char*>{
                            context_->GetNodeName(),
                            (std::string("HP L0C overflow (pingpong): mDeal=") + std::to_string(mDealB) +
                             ", D=" + std::to_string(dTile) + ", need=" +
                             std::to_string(static_cast<uint64_t>(2) * mDealB * dTile * ARCH31_FP16_BYTES * 2ULL) +
                             " > l0c=" + std::to_string(l0cSizeB))
                                .c_str()})),
                    return ge::GRAPH_FAILED);
        OP_LOGI(context_->GetNodeName(),
                "HP: code=%u prefill=%d D=%u l0b=%llu l0c=%llu ub=%llu "
                "-> mDeal=%u nDeal=%u kQK=%u kV=%u",
                hpCode, static_cast<int32_t>(hpPrefill), dTile, static_cast<unsigned long long>(l0bSizeB),
                static_cast<unsigned long long>(l0cSizeB), static_cast<unsigned long long>(ubSizeB), mBaseDealOut,
                nBaseOut, kTileQKOut, kTileVOut);
    }
    tilingDataPtr->innerSplitParams.set_mBaseSize(mBase);
    tilingDataPtr->innerSplitParams.set_s2BaseSize(nBaseOut);
    tilingDataPtr->innerSplitParams.set_dTile(dTile);
    tilingDataPtr->innerSplitParams.set_kTileQK(kTileQKOut);
    tilingDataPtr->innerSplitParams.set_nTileV(nTileVOut);
    tilingDataPtr->innerSplitParams.set_mBaseDeal(mBaseDealOut);
    tilingDataPtr->innerSplitParams.set_needSplitD(needSplitD ? 1 : 0);
    tilingDataPtr->innerSplitParams.set_kTileV(kTileVOut);

    uint64_t mm1ResSize = 0;
    uint64_t mm2ResSize = 0;
    if (hpPrefill) {
        mm1ResSize = static_cast<uint64_t>(B) * H * S1 * Arch31S2StridePad(S2) * 2;
        mm2ResSize = static_cast<uint64_t>(B) * H * S1 * D * 2;
    } else if (inputLayout == "TND") {
        uint64_t sumS1S2 = 0;
        uint64_t sumS1 = 0;
        AccumulateTndSums(seqQVals, seqQNum, seqKvVals, sumS1S2, sumS1);
        mm1ResSize = static_cast<uint64_t>(H) * sumS1S2;
        mm2ResSize = static_cast<uint64_t>(H) * sumS1 * D;
        if (mm1ResSize > ARCH31_TND_WS_INT32_MAX || mm2ResSize > ARCH31_TND_WS_INT32_MAX) {
            REPORT_PREDEFINED_ERR_MSG(
                "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                (std::vector<const char*>{
                    context_->GetNodeName(),
                    (std::string("arch31 TND workspace element count overflow: mm1ResSize=") +
                     std::to_string(mm1ResSize) + ", mm2ResSize=" + std::to_string(mm2ResSize) + " exceeds INT32_MAX")
                        .c_str()}));
            return ge::GRAPH_FAILED;
        }
    } else {
        mm1ResSize = static_cast<uint64_t>(B) * H * S1 * Arch31S2StridePad(S2);
        mm2ResSize = static_cast<uint64_t>(B) * H * S1 * D;
    }
    tilingDataPtr->workspaceParams.set_mm1ResSize(static_cast<uint32_t>(mm1ResSize));
    tilingDataPtr->workspaceParams.set_mm2ResSize(static_cast<uint32_t>(mm2ResSize));
    tilingDataPtr->workspaceParams.set_fdAccumOutSize(0);
    tilingDataPtr->workspaceParams.set_fdLogSumExpSize(0);

    uint32_t* outerBN2EndPtr = tilingDataPtr->outerSplitParams.get_bN2End();
    uint32_t* outerGS1EndPtr = tilingDataPtr->outerSplitParams.get_gS1End();
    uint32_t* outerS2EndPtr = tilingDataPtr->outerSplitParams.get_s2End();
    for (uint32_t i = 0; i < FIA_MAX_AIC_CORE_NUM; i++) {
        outerBN2EndPtr[i] = 0;
        outerGS1EndPtr[i] = 0;
        outerS2EndPtr[i] = 0;
    }

    tilingDataPtr->fdParams.set_numOfFdHead(0);
    tilingDataPtr->fdParams.set_reserved(hpCode);
    tilingDataPtr->fdParams.set_gS1BaseSizeOfFd(0);
    tilingDataPtr->fdParams.set_usedVecNumOfFd(0);
    uint32_t* fdBN2IdxOfFdHead = tilingDataPtr->fdParams.get_bN2IdxOfFdHead();
    uint32_t* fdGS1IdxOfFdHead = tilingDataPtr->fdParams.get_gS1IdxOfFdHead();
    uint32_t* fdS2SplitNumOfFdHead = tilingDataPtr->fdParams.get_s2SplitNumOfFdHead();
    uint32_t* fdS2SplitStartIdxOfCore = tilingDataPtr->fdParams.get_s2SplitStartIdxOfCore();
    uint32_t* fdGS1SplitNumOfFdHead = tilingDataPtr->fdParams.get_gS1SplitNumOfFdHead();
    uint32_t* fdGS1LastPartSizeOfFdHead = tilingDataPtr->fdParams.get_gS1LastPartSizeOfFdHead();
    uint32_t* fdGS1IdxEndOfFdHead = tilingDataPtr->fdParams.get_gS1IdxEndOfFdHead();
    uint32_t* fdGS1IdxEndOfFdHeadSplit = tilingDataPtr->fdParams.get_gS1IdxEndOfFdHeadSplit();
    for (uint32_t i = 0; i < FIA_MAX_AIC_CORE_NUM; i++) {
        fdBN2IdxOfFdHead[i] = 0;
        fdGS1IdxOfFdHead[i] = 0;
        fdS2SplitNumOfFdHead[i] = 0;
        fdS2SplitStartIdxOfCore[i] = 0;
        fdGS1SplitNumOfFdHead[i] = 0;
        fdGS1LastPartSizeOfFdHead[i] = 0;
    }
    for (uint32_t i = 0; i < FIA_MAX_AIC_CORE_NUM * 2; i++) {
        fdGS1IdxEndOfFdHead[i] = 0;
        fdGS1IdxEndOfFdHeadSplit[i] = 0;
    }
    tilingDataPtr->prefixParams.set_prefixMaxLen(0);
    tilingDataPtr->prefixParams.set_prefixLen(0);
    tilingDataPtr->prefixParams.set_prefixFlag(false);
    tilingDataPtr->pseParams.set_pseShiftFlag(0);
    tilingDataPtr->pseParams.set_pseShiftByBatch(0);
    tilingDataPtr->pseParams.set_pseShiftS1(0);
    tilingDataPtr->pseParams.set_pseShiftS2(0);
    tilingDataPtr->leftPaddingParams.set_qPaddingFlag(0);
    tilingDataPtr->leftPaddingParams.set_kvPaddingFlag(0);
    tilingDataPtr->postquantParams.set_isPerChnOut(0);
    tilingDataPtr->postquantParams.set_isOutQuantTypeBf16(0);
    tilingDataPtr->kvStrideParams.set_keyBnStride(0);
    tilingDataPtr->kvStrideParams.set_valueBnStride(0);
    tilingDataPtr->kvStrideParams.set_keyRopeBnStride(0);

    // 8. 成员对象序列化回写 context
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context_->GetNodeName(), "GetRawTilingData returned nullptr"})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingData_.GetDataSize(),
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        context_->GetNodeName(),
                        (std::string("RawTilingData capacity ") + std::to_string(rawTilingData->GetCapacity()) +
                         " < tiling data size " + std::to_string(tilingData_.GetDataSize()))
                            .c_str()})),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingNonQuantArch31::CalcTilingParams(uint32_t& mBase, uint32_t& nBase, uint32_t& dInner,
                                                          bool& needSplitD, uint32_t& dTile, uint32_t& kTileQK,
                                                          uint32_t& nTileV, uint32_t& mBaseDeal, uint32_t& kTileV)
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);

    auto queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto valueShape = context_->GetInputShape(2)->GetStorageShape();
    std::string layoutStr2 = "BNSD";
    uint32_t headsAttr2 = 0;
    uint32_t kvHeadsAttr2 = 0;
    auto attrs2 = context_->GetAttrs();
    if (attrs2 != nullptr) {
        auto layoutPtr = attrs2->GetStr(ATTR_INPUT_LAYOUT_INDEX);
        if (layoutPtr != nullptr) {
            layoutStr2 = std::string(layoutPtr);
        }
        auto headsPtr = attrs2->GetAttrPointer<int32_t>(ATTR_N_INDEX);
        if (headsPtr != nullptr) {
            headsAttr2 = static_cast<uint32_t>(*headsPtr);
        }
        auto kvHeadsPtr = attrs2->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
        if (kvHeadsPtr != nullptr) {
            kvHeadsAttr2 = static_cast<uint32_t>(*kvHeadsPtr);
        }
    }
    Arch31Dims dims2;
    const int64_t* seqQVals2 = nullptr;
    const int64_t* seqKvVals2 = nullptr;
    uint32_t seqQNum2 = 0;
    uint32_t seqKvNum2 = 0;
    if (layoutStr2 == "TND" && !LoadTndSeqPtrs(context_, seqQVals2, seqQNum2, seqKvVals2, seqKvNum2)) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "actual_seq_lengths/actual_seq_lengths_kv");
        return ge::GRAPH_FAILED;
    }
    if (!ParseLayoutDimsArch31(queryShape, keyShape, valueShape, layoutStr2, headsAttr2, kvHeadsAttr2, dims2, seqQVals2,
                               seqQNum2, seqKvVals2, seqKvNum2)) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      (std::string("CalcTilingParams: layout ") + layoutStr2 +
                                       " shape parse failed: query dimNum=" + std::to_string(queryShape.GetDimNum()) +
                                       ", key dimNum=" + std::to_string(keyShape.GetDimNum()) +
                                       ", value dimNum=" + std::to_string(valueShape.GetDimNum()))
                                          .c_str()}));
        return ge::GRAPH_FAILED;
    }
    uint32_t D = dims2.dQ;
    if (D == 0) {
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context_->GetNodeName(), "CalcTilingParams: D is 0"}));
        return ge::GRAPH_FAILED;
    }
    uint64_t l0cPingPongUnsplit = static_cast<uint64_t>(D) * ARCH31_FP16_BYTES * 2;
    uint32_t mUnsplitL0CCap = static_cast<uint32_t>(l0cSize / l0cPingPongUnsplit);
    bool isUnsplitCritical = (static_cast<uint64_t>(mUnsplitL0CCap) * l0cPingPongUnsplit == l0cSize);
    uint32_t mUnsplitL0CSafe = (D >= 512 && isUnsplitCritical) ? (mUnsplitL0CCap / 2) : mUnsplitL0CCap;
    if (mUnsplitL0CSafe == 0) {
        mUnsplitL0CSafe = 1;
    }

    uint32_t dTileHalf = D / 2;
    uint32_t mSplitL0CCap = 0;
    bool isSplitCritical = false;
    uint32_t mSplitL0CSafe = 0;
    if (dTileHalf > 0) {
        uint64_t l0cPingPongSplit = static_cast<uint64_t>(dTileHalf) * ARCH31_FP16_BYTES * 2;
        mSplitL0CCap = static_cast<uint32_t>(l0cSize / l0cPingPongSplit);
        isSplitCritical = (static_cast<uint64_t>(mSplitL0CCap) * l0cPingPongSplit == l0cSize);
        mSplitL0CSafe = (D >= 512 && isSplitCritical) ? (mSplitL0CCap / 2) : mSplitL0CCap;
        if (mSplitL0CSafe == 0) {
            mSplitL0CSafe = 1;
        }
    }

    uint32_t nDealMaxUnsplit = static_cast<uint32_t>(l0bSize / (static_cast<uint64_t>(D) * ARCH31_FP16_BYTES));
    uint32_t nDealMaxSplit = 0;
    if (dTileHalf > 0) {
        nDealMaxSplit = static_cast<uint32_t>(l0bSize / (static_cast<uint64_t>(dTileHalf) * ARCH31_FP16_BYTES * 2)) * 2;
    }

    auto ubMUpper = [&](uint32_t nDealMax) -> uint32_t { return Arch31UbMUpper(D, nDealMax); };
    uint32_t mUBcapUnsplit = ubMUpper(nDealMaxUnsplit);
    uint32_t mUBcapSplit = ubMUpper(nDealMaxSplit);
    if (mUBcapUnsplit == 0) {
        mUBcapUnsplit = 1;
    }
    if (mUBcapSplit == 0) {
        mUBcapSplit = 1;
    }

    uint32_t mBaseUnsplit = std::min(mUnsplitL0CSafe, mUBcapUnsplit);
    uint32_t mBaseSplit = std::min(mSplitL0CSafe, mUBcapSplit);

    needSplitD = (D >= 512) && (mBaseSplit > mBaseUnsplit);

    mBaseDeal = needSplitD ? mBaseSplit : mBaseUnsplit;
    dTile = needSplitD ? dTileHalf : D;
    kTileQK = dTile / 2;
    if (kTileQK < ARCH31_CUBE_K) {
        kTileQK = ARCH31_CUBE_K;
    }
    nTileV = dTile;
    uint32_t nDealMax = needSplitD ? nDealMaxSplit : nDealMaxUnsplit;
    if (nTileV == 0) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(), "CalcTilingParams: nTileV is 0, cannot derive kTileV"}));
        return ge::GRAPH_FAILED;
    }
    uint64_t kTileVRaw = l0bSize / (static_cast<uint64_t>(nTileV) * ARCH31_FP16_BYTES * 2);
    kTileV = static_cast<uint32_t>(kTileVRaw / ARCH31_CUBE_K) * ARCH31_CUBE_K;
    if (kTileV == 0) {
        kTileV = ARCH31_CUBE_K;
    }

    uint32_t cubeN = (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3003) ? 16U : 8U;
    nDealMax = (nDealMax / cubeN) * cubeN;
    if (nDealMax == 0) {
        nDealMax = cubeN;
    }
    if (nDealMax > ARCH31_N_BASE_UB) {
        nDealMax = ARCH31_N_BASE_UB;
    }
    nBase = nDealMax;
    mBase = mBaseDeal;
    dInner = D;

    return ge::GRAPH_SUCCESS;
}

uint64_t FiaTilingNonQuantArch31::CalcTilingKey(uint32_t sparseMode, uint32_t layout, bool attenMaskTensorFlag,
                                                uint32_t hpCode)
{
    // 19 位编码（前缀首位=6）：
    //  [18]    = arch flag (6) -> FIA_ARCH31_FLAG_TILING
    //  [17]    = Q type (FP16=0)
    //  [16]    = KV type (FP16=0)
    //  [15]    = OUT type (FP16=0)
    //  [14-13] = layout (BNSD=00, BSND=01, BSH=10, TND=11)
    //  [12]    = reserved
    //  [11-8]  = smBits
    //  [7-0]   = reserved
    uint32_t smBits = (sparseMode == 1 || (attenMaskTensorFlag && sparseMode != 9)) ? 1 : sparseMode;
    uint64_t key = FIA_ARCH31_FLAG_TILING;
    key |= static_cast<uint64_t>(layout & 0x3) << 13;
    key |= static_cast<uint64_t>(smBits & 0xF) << 8;
    key |= static_cast<uint64_t>(hpCode & 0xFFU);
    return key;
}

ge::graphStatus FiaTilingNonQuantArch31::CalcWorkspaceSize(uint64_t& workspaceSize, uint32_t mBase, uint32_t nBase,
                                                           uint32_t dInner)
{
    (void)mBase;
    (void)nBase;
    (void)dInner;
    auto queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto valueShape = context_->GetInputShape(2)->GetStorageShape();
    std::string layoutStr3 = "BNSD";
    uint32_t headsAttr3 = 0;
    uint32_t kvHeadsAttr3 = 0;
    auto attrs3 = context_->GetAttrs();
    if (attrs3 != nullptr) {
        auto layoutPtr = attrs3->GetStr(ATTR_INPUT_LAYOUT_INDEX);
        if (layoutPtr != nullptr) {
            layoutStr3 = std::string(layoutPtr);
        }
        auto headsPtr = attrs3->GetAttrPointer<int32_t>(ATTR_N_INDEX);
        if (headsPtr != nullptr) {
            headsAttr3 = static_cast<uint32_t>(*headsPtr);
        }
        auto kvHeadsPtr = attrs3->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
        if (kvHeadsPtr != nullptr) {
            kvHeadsAttr3 = static_cast<uint32_t>(*kvHeadsPtr);
        }
    }
    Arch31Dims dims3;
    const int64_t* seqQVals3 = nullptr;
    const int64_t* seqKvVals3 = nullptr;
    uint32_t seqQNum3 = 0;
    uint32_t seqKvNum3 = 0;
    if (layoutStr3 == "TND" && !LoadTndSeqPtrs(context_, seqQVals3, seqQNum3, seqKvVals3, seqKvNum3)) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "actual_seq_lengths/actual_seq_lengths_kv");
        return ge::GRAPH_FAILED;
    }
    if (!ParseLayoutDimsArch31(queryShape, keyShape, valueShape, layoutStr3, headsAttr3, kvHeadsAttr3, dims3, seqQVals3,
                               seqQNum3, seqKvVals3, seqKvNum3)) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      (std::string("CalcWorkspaceSize: layout ") + layoutStr3 +
                                       " shape parse failed: query dimNum=" + std::to_string(queryShape.GetDimNum()) +
                                       ", key dimNum=" + std::to_string(keyShape.GetDimNum()) +
                                       ", value dimNum=" + std::to_string(valueShape.GetDimNum()))
                                          .c_str()}));
        return ge::GRAPH_FAILED;
    }
    uint32_t B = dims3.b;
    uint32_t H = dims3.heads;
    uint32_t S1 = dims3.s1;
    uint32_t D = dims3.dQ;
    uint32_t S2 = dims3.s2;
    bool hpPrefillParityWs = false;
    do {
        if (layoutStr3 != "BNSD" || B != 1 || H != 8 || S1 != 64 || (S2 != 2048 && S2 != 576)) {
            break;
        }
        if (dims3.kvHeads != 2) {
            break;
        }
        hpPrefillParityWs = true;
    } while (false);
    uint64_t mm1ResSize = 0;
    uint64_t softmaxPFp16Size = 0;
    uint64_t mm2ResSize = 0;
    if (layoutStr3 == "TND") {
        uint64_t sumS1S2 = 0;
        uint64_t sumS1 = 0;
        AccumulateTndSums(seqQVals3, seqQNum3, seqKvVals3, sumS1S2, sumS1);
        mm1ResSize = static_cast<uint64_t>(H) * sumS1S2 * ARCH31_FP16_BYTES;
        softmaxPFp16Size = mm1ResSize;
        mm2ResSize = static_cast<uint64_t>(H) * sumS1 * D * ARCH31_FP16_BYTES;
    } else {
        uint32_t parityMul = hpPrefillParityWs ? 2U : 1U;
        mm1ResSize = static_cast<uint64_t>(B) * H * S1 * Arch31S2StridePad(S2) * ARCH31_FP16_BYTES * parityMul;
        softmaxPFp16Size = mm1ResSize;
        mm2ResSize = static_cast<uint64_t>(B) * H * S1 * D * ARCH31_FP16_BYTES * parityMul;
    }
    workspaceSize = mm1ResSize + softmaxPFp16Size + mm2ResSize;
    workspaceSize = (workspaceSize + 31) & ~31ULL;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
