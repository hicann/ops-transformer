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
 * \file quant_flash_attention_score_grad_tiling_normal_regbase.cpp
 * \brief
 */
#include "quant_flash_attention_score_grad_tiling_normal_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "err/ops_err.h"
#include "../quant_flash_attn_grad_tiling.h"

using namespace Ops::Transformer::OpTiling;

using namespace optiling::QuantFag;
namespace optiling {
namespace QuantFag {

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::GetShapeAttrsInfo()
{
    fBaseParams.isDeterministic = true;
    const gert::StorageShape *queryShape = context_->GetInputShape(QUERY_IDX); // [B, N2, G, S1, D]
    const gert::StorageShape *keyShape = context_->GetInputShape(KEY_IDX);     // [B, N2, 1, S2, D]
    const gert::StorageShape *valueShape = context_->GetInputShape(VALUE_IDX); // [B, N2, 1, S2, D_V]

    const char *inputLayoutQ = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_Q));
    const char *inputLayoutKV = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_KV));
    if (inputLayoutQ == nullptr && inputLayoutKV == nullptr) {
        inputLayoutQ = "BSND";
    } else if (inputLayoutQ == nullptr || inputLayoutKV == nullptr || strcmp(inputLayoutQ, inputLayoutKV) != 0) {
        OP_LOGE(context_, "inputLayoutQ and inputLayoutKV must be same.");
        return ge::GRAPH_FAILED;
    }
    int64_t headNum = 0;
    if (strcmp(inputLayoutQ, "BSND") == 0) {
        // q shape = [B, S, N, D]
        headNum = queryShape->GetStorageShape().GetDim(2);
    } else if (strcmp(inputLayoutQ, "BNSD") == 0) {
        // q shape = [B, N, S, D]
        headNum = queryShape->GetStorageShape().GetDim(1);
    } else {
        OP_LOGE(context_, "Invalid layout_q: %s, only BSND/BNSD supported", inputLayoutQ);
        return ge::GRAPH_FAILED;
    }

    if (strcmp(inputLayoutQ, "BNSD") == 0) {
        OP_LOGD(context_, "inputLayout == BNSD queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_BN2GS2D;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_1) / keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        OP_LOGD(context_, "inputLayout == BNSD queryShape", "%ld, %ld, %ld, %ld,",
                queryShape->GetStorageShape().GetDim(INPUT_DIM_0), queryShape->GetStorageShape().GetDim(INPUT_DIM_1),
                queryShape->GetStorageShape().GetDim(INPUT_DIM_2), queryShape->GetStorageShape().GetDim(INPUT_DIM_3));
    } else {
        OP_LOGD(context_, "inputLayout == BSND queryShape");
        // inputLayout = "BSND"
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
    }

    fBaseParams.n1 = fBaseParams.n2 * fBaseParams.g;

    auto ret = ProcessOptionalInput(context_, fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ret;
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::GetPlatformInfo()
{
    uint32_t coreNum = CORE_INIT_NUM; // 40 is init core num

    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const QuantFlashAttnGradCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr,
                    OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "compile_info is null"),
                    return ge::GRAPH_FAILED);
        npuArch = compileInfoPtr->npuArch;
        fBaseParams.coreNum = compileInfoPtr->aivNum;
        fBaseParams.aicNum = compileInfoPtr->aicNum;
        fBaseParams.ubSize = compileInfoPtr->ubSize;
        fBaseParams.l1Size = compileInfoPtr->l1Size;
        fBaseParams.l0aSize = compileInfoPtr->l0aSize;
        fBaseParams.l0cSize = compileInfoPtr->l0cSize;
        fBaseParams.l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        npuArch = ascendcPlatform.GetCurNpuArch();
        coreNum = ascendcPlatform.GetCoreNumAiv();
        fBaseParams.coreNum = coreNum;
        fBaseParams.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, fBaseParams.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, fBaseParams.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, fBaseParams.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, fBaseParams.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, fBaseParams.l2CacheSize);
    }
    OP_CHECK_IF((fBaseParams.coreNum == 0) || (fBaseParams.aicNum == 0),
                OP_LOGE(context_->GetNodeName(), "num of coreNum(aivNum) is %lu, num of aicNum is %lu.",
                        fBaseParams.coreNum, fBaseParams.aicNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fBaseParams.ubSize <= 0, OP_LOGE(context_->GetNodeName(), "ubSize is invalid."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool QuantFlashAttentionScoreGradTilingNormalRegbase::IsCapable()
{
    // 基础模板 全部支持
    if (npuArch == NpuArch::DAV_3510) {
        OP_LOGD(context_, "QuantFlashAttentionScoreGradTilingNormalRegbase hit");
        return true;
    }
    return false;
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::DoOpTiling()
{
    fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
    DoSplit();
    auto ret = DoSparse();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = InitTilingData();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    DoPreTiling();
    DoPostTiling();
    DetermineMode(fBaseParams);
    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttentionScoreGradTilingNormalRegbase::DoSplit()
{
    fBaseParams.s1CvRatio = S1CV_RATIO_DEFAULT;
    fBaseParams.s2CvRatio = S2CV_RATIO_DEFAULT;
    std::tuple<uint32_t, uint32_t, uint32_t> bestSplitRes = FuzzyForBestSplit();
    int64_t s1Inner = std::get<0>(bestSplitRes);
    int64_t s1CvInner =
        s1Inner * fBaseParams.s1CvRatio > fBaseParams.s1 ? fBaseParams.s1 : s1Inner * fBaseParams.s1CvRatio;
    int64_t s1Outer = (fBaseParams.s1 + s1CvInner - 1) / s1CvInner;
    int64_t s1TailTmp = fBaseParams.s1 % s1Inner;
    int64_t s1CvTailTmp = fBaseParams.s1 % s1CvInner;
    fBaseParams.s1CvTail = s1CvTailTmp == 0 ? s1CvInner : s1CvTailTmp;
    fBaseParams.s1Inner = s1Inner;
    fBaseParams.s1CvInner = s1CvInner;
    fBaseParams.s1Outer = s1Outer;

    int64_t s2Inner = std::get<1>(bestSplitRes);
    int64_t cvS2Inner =
        s2Inner * fBaseParams.s2CvRatio > fBaseParams.s2 ? fBaseParams.s2 : s2Inner * fBaseParams.s2CvRatio;
    int64_t s2Outer = (fBaseParams.s2 + cvS2Inner - 1) / cvS2Inner;
    int64_t s2TailTmp = fBaseParams.s2 % s2Inner;
    int64_t s2CvTailTmp = fBaseParams.s2 % cvS2Inner;
    fBaseParams.s2CvTail = s2CvTailTmp == 0 ? cvS2Inner : s2CvTailTmp;
    fBaseParams.s2Outer = s2Outer;
    fBaseParams.cvS2Inner = cvS2Inner;
    fBaseParams.s2Inner = s2Inner;

    uint32_t sfmgdInner = std::get<2>(bestSplitRes);
    fBaseParams.sfmgdInner = sfmgdInner;
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::DoSparse()
{
    fBaseParams.sparseType = GetSparseType(); // 非确定性计算下获取sparseType
    fBaseParams.deterSparseType = GetDeterSparseTilingKey();
    CalcleDeterParam();
    fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
    if (fBaseParams.isSparse) {
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
            fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
            GetSparsePrefixBlockInfo();
        } else {
            GetSparseBlockInfo();
        }
    } else {
        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];
        int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;
        int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
        int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

        fBaseParams.blockOuter = blockOuter;
        fBaseParams.blockFactor = blockFactor;
        fBaseParams.maxValidBBLen = blockFactor;

        for (int64_t i = 0; i < blockOuter; i++) {
            blockStarts[i] = blockFactor * i;
            blockEnds[i] = std::min(blockFactor * (i + 1), fusedOuter);
        }
        for (uint32_t i = static_cast<uint32_t>(blockOuter); i < CORE_LIST_NUM; i++) {
            blockStarts[i] = 0;
            blockEnds[i] = 0;
        }

        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));
    }
    // each bit init 1
    std::fill(std::begin(fBaseParams.dqIsNeedDeter), std::end(fBaseParams.dqIsNeedDeter), static_cast<uint64_t>(-1));
    std::fill(std::begin(fBaseParams.dkDvIsNeedDeter), std::end(fBaseParams.dkDvIsNeedDeter),
              static_cast<uint64_t>(-1));
    if (fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_OLD)) {
        GetIsDeterArr();
    }
    return ge::GRAPH_SUCCESS;
}

bool QuantFlashAttentionScoreGradTilingNormalRegbase::CheckSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft,
                                                                              int64_t s2IdxRight, int64_t bIdx,
                                                                              int64_t blockIdx)
{
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        int64_t s2IgnoredEndLen = static_cast<int64_t>(fBaseParams.s1) -
                                  static_cast<int64_t>(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1));
        int64_t s2EndLen = static_cast<int64_t>(fBaseParams.s2) > s2IgnoredEndLen ?
                               static_cast<int64_t>(fBaseParams.s2) - s2IgnoredEndLen :
                               0;
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
            fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
            int64_t curBIdx = blockIdx / (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
            s2EndLen = std::min(std::max(s2EndLen, static_cast<int64_t>(fBaseParams.prefixN[curBIdx])),
                                static_cast<int64_t>(fBaseParams.s2));
        }
        bool isValid = s2IdxLeft < s2EndLen;
        return isValid;
    } else {
        int64_t s2SparseLeft = std::max(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * s1oDimIdx - fBaseParams.s1Token,
                                        static_cast<int64_t>(0));
        s2SparseLeft = AlignTo(s2SparseLeft, ALIGN64);
        int64_t s2SparseRight = AlignTo(
            std::min(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), fBaseParams.s1) + fBaseParams.s2Token,
            static_cast<int64_t>(64));
        s2SparseRight = std::min(s2SparseRight, fBaseParams.s2);
        bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
        return isValid;
    }
}

bool QuantFlashAttentionScoreGradTilingNormalRegbase::IsValid(int64_t blockIdx)
{
    int64_t gDimTail = blockIdx % (fBaseParams.s1Outer * fBaseParams.s2Outer);
    int64_t s2oDimIdx = gDimTail / fBaseParams.s1Outer;
    int64_t s1oDimIdx = gDimTail % fBaseParams.s1Outer;
    int64_t s2IdxLeft = s2oDimIdx * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT;
    int64_t s2IdxRight = std::min((s2oDimIdx + 1) * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT, fBaseParams.s2);
    if (fBaseParams.attenMaskOptional != EMPTY_TENSOR) {
        return CheckSparseLeftAndRight(s1oDimIdx, s2IdxLeft, s2IdxRight, static_cast<int64_t>(0), blockIdx);
    }
    return true;
}

uint32_t QuantFlashAttentionScoreGradTilingNormalRegbase::GetDeterSparseTilingKey()
{
    if (!fBaseParams.isDeterministic) {
        return static_cast<uint32_t>(DeterSparseType::NO_DETER);
    }

    if (!fBaseParams.isSparse || (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK)) ||
        (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
         fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token >= fBaseParams.s2)) {
        return static_cast<uint32_t>(DeterSparseType::DETER_DENSE);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                fBaseParams.s1Token >= fBaseParams.s1 &&
                (fBaseParams.s2Token > NEGATIVE_128 && fBaseParams.s2Token <= 0)) ||
               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
                fBaseParams.isS1S2Same)) {
        return static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
               // RIGHT_DOWN_CAUSAL场景和Band类似，直接走Band分支
               fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
               fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
        return static_cast<uint32_t>(DeterSparseType::DETER_BAND);
    }
    return static_cast<uint32_t>(DeterSparseType::DETER_OLD);
}

uint8_t QuantFlashAttentionScoreGradTilingNormalRegbase::GetSparseType()
{
    // DENSE: 1）非sparse；2）ALL_MASK；3）NO_MASK & preToken>=Sq & nextToken>=Skv
    bool denseCondition = !fBaseParams.isSparse ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token >= fBaseParams.s2);

    bool casualCondition = false;
    bool bandCondition = false;
    // CASUAL: 1）LEFT_UP_CASUAL；2）RIGHT_DOWN_CASUAL；3）NO_MASK & preToken>=Sq & nextToken=0；4）BAND &
    // preToken>=Sq & nextToken=0
    casualCondition = (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) &&
                       fBaseParams.s1 <= fBaseParams.s2) ||
                      (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                       fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0) ||
                      (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
                       fBaseParams.s1 <= fBaseParams.s2) ||
                      (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) &&
                       fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0);

    // BAND: 1）NO_MASK剩余场景；2）BAND剩余场景；3）LEFT_UP_CAUSAL剩余场景；4）RIGHT_DOWN_CAUSAL剩余场景
    bandCondition = fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
                    fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
                    fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
                    fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL);

    if (denseCondition) {
        return static_cast<uint8_t>(SparseType::DENSE);
    } else if (casualCondition) {
        return static_cast<uint8_t>(SparseType::CASUAL);
    } else if (bandCondition) {
        return static_cast<uint8_t>(SparseType::BAND);
    } else {
        // 超L2优化暂不支持的sparse场景
        return static_cast<uint8_t>(SparseType::UNSUPPORTED);
    }
}

void QuantFlashAttentionScoreGradTilingNormalRegbase::CalcleDeterParam()
{
    if (!fBaseParams.isDeterministic ||
        fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_OLD)) {
        return;
    }
    // EOD场景确定性计算需要使用真实的b
    int64_t oriBsize = fBaseParams.b;
    fBaseParams.b -= fBaseParams.tailZeroCount;

    int64_t cubebaseM = fBaseParams.s1Inner * fBaseParams.s1CvRatio;
    int64_t cubebaseN = fBaseParams.s2Inner * fBaseParams.s2CvRatio;
    uint8_t deterTilingSplitMode = (cubebaseM == cubebaseN ? 0 : (cubebaseM > cubebaseN ? 2 : 1));
    int64_t s1Outer{fBaseParams.s1Outer};
    int64_t s2Outer{fBaseParams.s2Outer};
    int64_t s1Inner{fBaseParams.s1Inner};
    int64_t s2Inner{fBaseParams.s2Inner};
    bool needChangeSplitItemMode2 =
        (deterTilingSplitMode == 2) &&
        (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_DENSE));
    bool needChangeSplitItemMode1 =
        (deterTilingSplitMode == 1) &&
        (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_DENSE));
    // 若是256 * 128或64 * 128切分，则
    if (needChangeSplitItemMode2) {
        fBaseParams.s2Inner = fBaseParams.s2Inner * NUM_TWO;
        fBaseParams.s2Outer = CeilDivideBy(s2Outer, static_cast<int64_t>(NUM_TWO));
    }
    if (needChangeSplitItemMode1) {
        fBaseParams.s1Inner = fBaseParams.s1Inner * NUM_TWO;
        fBaseParams.s1Outer = CeilDivideBy(s1Outer, static_cast<int64_t>(NUM_TWO));
    }
    if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
        fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL)) {
        CalcleCausalDeterParam(fBaseParams);
    } else if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
               fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
        CalcleBandDeterParam(fBaseParams);
    }
    if (needChangeSplitItemMode1 || needChangeSplitItemMode2) {
        fBaseParams.s1Outer = s1Outer;
        fBaseParams.s2Outer = s2Outer;
        fBaseParams.s1Inner = s1Inner;
        fBaseParams.s2Inner = s2Inner;
        fBaseParams.deterMaxRound *= NUM_TWO;
    }
    // 还原eod
    fBaseParams.b = oriBsize;
}

void QuantFlashAttentionScoreGradTilingNormalRegbase::GetIsDeterArr()
{
    std::array<int64_t, CORE_LIST_NUM> dqOffset;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffset;
    std::array<int64_t, CORE_LIST_NUM> dqOffsetpre;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffsetpre;
    std::array<int64_t, CORE_LIST_NUM> loopIdx;
    bool dqNeedDeterpre = false;
    bool dkDvNeedDeterpre = false;
    int64_t calcNum = 0;
    std::fill(std::begin(loopIdx), std::end(loopIdx), static_cast<int64_t>(0));
    while (calcNum < fBaseParams.maxValidBBLen) {
        for (uint16_t cBlockIdx = 0; cBlockIdx < fBaseParams.blockOuter; cBlockIdx++) {
            while (!IsValid(fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx]) &&
                   (fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx] < fBaseParams.blockEnds[cBlockIdx])) {
                loopIdx[cBlockIdx]++;
            }
            if (fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx] >= fBaseParams.blockEnds[cBlockIdx]) {
                dqOffset[cBlockIdx] = OUTINDEX;
                dkDvOffset[cBlockIdx] = OUTINDEX;
                continue;
            }
            int64_t validBlockIdx = fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx];
            GetOffset(fBaseParams, dqOffset[cBlockIdx], dkDvOffset[cBlockIdx], validBlockIdx);
            loopIdx[cBlockIdx]++;
        }
        JudgeIsNeedDeter(fBaseParams, dqOffset, dkDvOffset, dqOffsetpre, dkDvOffsetpre, calcNum,
                         fBaseParams.noNeedDeter, dqNeedDeterpre, dkDvNeedDeterpre);
        calcNum++;
    }
}

uint64_t QuantFlashAttentionScoreGradTilingNormalRegbase::DoPreSfmgTiling()
{
    uint32_t valueDAlign = fBaseParams.sfmgdInner;

    int64_t normalAxisSize = 0;
    normalAxisSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1;

    int32_t inputSize = FP16_BYTES;
    int32_t outDtypeSize = FP16_BYTES;
    // 计算单loop的计算量及loop次数, hifp8场景按128对齐, quantblock大小为128 * 4, 目前仅支持D <= 256
    int64_t singleLoopNBurstNum = 128;
    if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
        fBaseParams.queryType == ge::DT_HIFLOAT8) {
        inputSize = 1;
        outDtypeSize = FP16_BYTES;
    }
    uint32_t availUbSize = fBaseParams.ubSize - UB_RESERVE_SPACE;
    // valueDAlign * inputSize * sizeof(dtype) * 2 * 2 --  dy, y size is valueDAlign * inputSize
    // first 2 is dy + y total size, second 2 is double buffer, then get max split s1
    uint32_t sfmgDyBufferLen =
        availUbSize / (valueDAlign * (inputSize * 2 + outDtypeSize * 2) + 2 * 8 * FP32_BYTES) * valueDAlign * inputSize;
    uint32_t sfmgYBufferLen = availUbSize / (valueDAlign * (inputSize * 2 + outDtypeSize * 2) + 2 * 8 * FP32_BYTES) *
                              valueDAlign * outDtypeSize;
    uint32_t sfmgOutputBufferLen =
        availUbSize / (valueDAlign * (inputSize * 2 + outDtypeSize * 2) + 2 * 8 * FP32_BYTES) * 8 * FP32_BYTES;

    // 计算单核的计算量
    uint32_t sfmgUsedCoreNum = fBaseParams.blockOuter * 2;
    int64_t normalCoreSize = CeilCommon(normalAxisSize, sfmgUsedCoreNum);
    sfmgUsedCoreNum = CeilCommon(normalAxisSize, normalCoreSize);
    int64_t tailCoreSize = normalAxisSize - (sfmgUsedCoreNum - 1) * normalCoreSize;
    // 非fp8场景按照实际head dim的大小计算
    if (fBaseParams.queryType == ge::DT_FLOAT16 || fBaseParams.queryType == ge::DT_BF16) {
        singleLoopNBurstNum = sfmgDyBufferLen / inputSize / valueDAlign;
    }
    int64_t normalCoreLoopTimes = CeilCommon(normalCoreSize, singleLoopNBurstNum);
    int64_t normalCoreLastLoopNBurstNum = normalCoreSize - (normalCoreLoopTimes - 1) * singleLoopNBurstNum;
    int64_t tailCoreLoopTimes = CeilCommon(tailCoreSize, singleLoopNBurstNum);
    int64_t tailCoreLastLoopNBurstNum = tailCoreSize - (tailCoreLoopTimes - 1) * singleLoopNBurstNum;

    OP_LOGI("DoPreSfmgTiling",
            "DoPreSfmgTiling, sfmgUsedCoreNum = %d, ubsize = %d, valueDAlign = %d,"
            "normalAxisSize = %d, reals1percore = %d, sfmgDyBufferLen is %d, sfmgYBufferLen is %d, sfmgOutputBufferLen "
            "is %d."
            "singleLoopNBurstNum = %d, normalCoreLoopTimes is %d, normalCoreLastLoopNBurstNum is %d."
            "tailCoreLoopTimes = %d, tailCoreLastLoopNBurstNum is %d.",
            sfmgUsedCoreNum, availUbSize, valueDAlign, normalAxisSize, normalCoreSize, sfmgDyBufferLen, sfmgYBufferLen,
            sfmgOutputBufferLen, singleLoopNBurstNum, normalCoreLoopTimes, normalCoreLastLoopNBurstNum,
            tailCoreLoopTimes, tailCoreLastLoopNBurstNum);
    quantFagTilingData_->sfmg_used_core_num = sfmgUsedCoreNum;
    quantFagTilingData_->sfmg_dy_buffer_len = sfmgDyBufferLen;
    quantFagTilingData_->sfmg_y_buffer_len = sfmgYBufferLen;
    quantFagTilingData_->sfmg_output_buffer_len = sfmgOutputBufferLen;
    quantFagTilingData_->single_loop_nburst_num = singleLoopNBurstNum;
    quantFagTilingData_->normal_core_loop_times = normalCoreLoopTimes;
    quantFagTilingData_->tail_core_loop_times = tailCoreLoopTimes;
    quantFagTilingData_->normal_core_last_loop_nburst_num = normalCoreLastLoopNBurstNum;
    quantFagTilingData_->tail_core_last_loop_nburst_num = tailCoreLastLoopNBurstNum;
    quantFagTilingData_->normal_core_nburst_nums = normalCoreSize;
    quantFagTilingData_->tail_core_nburst_nums = tailCoreSize;
    quantFagTilingData_->normal_axis_size = normalAxisSize;
    return sfmgUsedCoreNum;
}

void QuantFlashAttentionScoreGradTilingNormalRegbase::DoPreTiling()
{
    uint64_t inputBufferLen = PRE_BUFFER_SIZE; // x / 8 + 2 * x + 32 = fBaseParams.ubSize
    uint64_t singleUBProcessNum = static_cast<uint64_t>(CAST_BUFFER_LEN) / 2;

    uint64_t maskSize = AlignTo(fBaseParams.dropMaskSize, static_cast<uint64_t>(BOOL_BLOCK_NUMS));
    uint64_t singleCoreNum = AlignTo(CeilDivideBy(maskSize, static_cast<uint64_t>(fBaseParams.blockOuter)),
                                     static_cast<uint64_t>(BOOL_BLOCK_NUMS));
    uint64_t maskUsedCoreNum = 0;
    fBaseParams.enablePreSfmg =
        (fBaseParams.queryType == ge::DT_HIFLOAT8) ||
        ((fBaseParams.queryType == ge::DT_BF16 || fBaseParams.queryType == ge::DT_FLOAT16) &&
         fBaseParams.d > static_cast<uint32_t>(ConstAxisTemplateNum::NUM64) &&
         fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM768) &&
         (fBaseParams.splitAxis == SplitAxisEnum::BN2GS1S2 || fBaseParams.splitAxis == SplitAxisEnum::BN2S2) &&
         !fBaseParams.isDeterministic && fBaseParams.dropoutIsDivisibleBy8);
    if (fBaseParams.enablePreSfmg) {
        maskUsedCoreNum = static_cast<uint64_t>(DoPreSfmgTiling());
    } else {
        maskUsedCoreNum = static_cast<uint64_t>(CeilDivideBy(maskSize, singleCoreNum));
    }
    OP_LOGI("DoPreTiling", "enablePreSfmg = %d, maskUsedCoreNum = %ld", fBaseParams.enablePreSfmg, maskUsedCoreNum);

    uint64_t tailCoreNum = maskSize - (maskUsedCoreNum - 1) * singleCoreNum;
    tailCoreNum = AlignTo(tailCoreNum, static_cast<uint64_t>(BOOL_BLOCK_NUMS));

    uint64_t singleCoreUBLoop = static_cast<uint64_t>(CeilDivideBy(singleCoreNum, singleUBProcessNum));
    uint64_t tailCoreUBLoop = static_cast<uint64_t>(CeilDivideBy(tailCoreNum, singleUBProcessNum));

    uint64_t singleCoreUBLastLoopNum =
        static_cast<uint64_t>(singleCoreNum - (singleCoreUBLoop - 1) * singleUBProcessNum);
    uint64_t tailCoreUBLastLoopNum = static_cast<uint64_t>(tailCoreNum - (tailCoreUBLoop - 1) * singleUBProcessNum);

    uint64_t qSize = fBaseParams.qSize;
    uint64_t kSize = fBaseParams.kSize;
    uint64_t vSize = fBaseParams.vSize;
    uint64_t qPreBlockFactor = (qSize + maskUsedCoreNum - 1) / maskUsedCoreNum;
    uint64_t qPreBlockTotal = (qSize + qPreBlockFactor - 1) / qPreBlockFactor;
    uint64_t qPreTailNumTmp = qSize % qPreBlockFactor;
    uint64_t qPreTailNum = qPreTailNumTmp == static_cast<uint64_t>(0) ? qPreBlockFactor : qPreTailNumTmp;

    uint64_t kPreBlockFactor = (kSize + maskUsedCoreNum - 1) / maskUsedCoreNum;
    uint64_t kPreBlockTotal = (kSize + kPreBlockFactor - 1) / kPreBlockFactor;
    uint64_t kPreTailNumTmp = kSize % kPreBlockFactor;
    uint64_t kPreTailNum = kPreTailNumTmp == static_cast<uint64_t>(0) ? kPreBlockFactor : kPreTailNumTmp;

    uint64_t vPreBlockFactor = (vSize + maskUsedCoreNum - 1) / maskUsedCoreNum;
    uint64_t vPreBlockTotal = (vSize + vPreBlockFactor - 1) / vPreBlockFactor;
    uint64_t vPreTailNumTmp = vSize % vPreBlockFactor;
    uint64_t vPreTailNum = vPreTailNumTmp == static_cast<uint64_t>(0) ? vPreBlockFactor : vPreTailNumTmp;

    uint64_t maskPreBlockTotal = fBaseParams.dropMaskSize;
    quantFagTilingData_->q_pre_block_factor = qPreBlockFactor;
    quantFagTilingData_->q_pre_block_total = qPreBlockTotal;

    quantFagTilingData_->q_pre_block_tail = qPreTailNum;

    quantFagTilingData_->k_pre_block_factor = kPreBlockFactor;
    quantFagTilingData_->k_pre_block_total = kPreBlockTotal;
    quantFagTilingData_->k_pre_block_tail = kPreTailNum;
    quantFagTilingData_->v_pre_block_factor = vPreBlockFactor;
    quantFagTilingData_->v_pre_block_total = vPreBlockTotal;
    quantFagTilingData_->v_pre_block_tail = vPreTailNum;
}

void QuantFlashAttentionScoreGradTilingNormalRegbase::DoPostTiling()
{
    uint64_t postUbBaseSize = REGBASE_POST_BASE * FP16_BYTES;
    uint64_t qPostBaseNum = REGBASE_POST_BASE;
    uint64_t qPostBlockTotal = static_cast<uint64_t>(fBaseParams.qSize);
    uint64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    uint64_t qPostTailNum = qPostTailNumTmp == static_cast<uint64_t>(0) ? qPostBaseNum : qPostTailNumTmp;
    uint64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - static_cast<uint64_t>(1)) / qPostBaseNum;
    uint64_t qPostBlockFactor = (qPostBlockOuterTotal + fBaseParams.blockOuter * AICV_RATIO_DEFAULT - 1) /
                                (fBaseParams.blockOuter * AICV_RATIO_DEFAULT);

    uint64_t kPostBaseNum = postUbBaseSize / FP16_BYTES;
    uint64_t kPostBlockTotal = static_cast<uint64_t>(fBaseParams.kSize);
    uint64_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    uint64_t kPostTailNum = kPostTailNumTmp == static_cast<uint64_t>(0) ? kPostBaseNum : kPostTailNumTmp;
    uint64_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - static_cast<uint64_t>(1)) / kPostBaseNum;
    uint64_t kPostBlockFactor = (kPostBlockOuterTotal + fBaseParams.blockOuter * AICV_RATIO_DEFAULT - 1) /
                                (fBaseParams.blockOuter * AICV_RATIO_DEFAULT);

    uint64_t vPostBaseNum = postUbBaseSize / FP16_BYTES;
    uint64_t vPostBlockTotal = static_cast<uint64_t>(fBaseParams.vSize);
    uint64_t vPostTailNumTmp = vPostBlockTotal % vPostBaseNum;
    uint64_t vPostTailNum = vPostTailNumTmp == static_cast<uint64_t>(0) ? vPostBaseNum : vPostTailNumTmp;
    uint64_t vPostBlockOuterTotal = (vPostBlockTotal + vPostBaseNum - static_cast<uint64_t>(1)) / vPostBaseNum;
    uint64_t vPostBlockFactor = (vPostBlockOuterTotal + fBaseParams.blockOuter * AICV_RATIO_DEFAULT - 1) /
                                (fBaseParams.blockOuter * AICV_RATIO_DEFAULT);
    quantFagTilingData_->q_post_block_factor = qPostBlockFactor;
    quantFagTilingData_->q_post_block_total = qPostBlockTotal;
    quantFagTilingData_->q_post_base_num = qPostBaseNum;
    quantFagTilingData_->q_post_tail_num = qPostTailNum;
    quantFagTilingData_->k_post_block_factor = kPostBlockFactor;
    quantFagTilingData_->k_post_block_total = kPostBlockTotal;
    quantFagTilingData_->k_post_base_num = kPostBaseNum;
    quantFagTilingData_->k_post_tail_num = kPostTailNum;
    quantFagTilingData_->v_post_block_factor = vPostBlockFactor;
    quantFagTilingData_->v_post_block_total = vPostBlockTotal;
    quantFagTilingData_->v_post_base_num = vPostBaseNum;
    quantFagTilingData_->v_post_tail_num = vPostTailNum;
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    size_t workspaceSize = 0;
    workspaceSize = RESERVED_WORKSPACE_SIZE;
    int64_t qSize =
        ((fBaseParams.b * fBaseParams.n1 - 1) * fBaseParams.s1 + AlignTo(fBaseParams.s1, ALIGN128)) * fBaseParams.d;
    int64_t kSize =
        ((fBaseParams.b * fBaseParams.n2 - 1) * fBaseParams.s2 + AlignTo(fBaseParams.s2, ALIGN128)) * fBaseParams.d;
    int64_t vSize =
        ((fBaseParams.b * fBaseParams.n2 - 1) * fBaseParams.s2 + AlignTo(fBaseParams.s2, ALIGN128)) * fBaseParams.d1;
    if (fBaseParams.queryType != ge::DT_FLOAT) {
        quantFagTilingData_->dq_work_space_offset = workspaceSize;
        // matmal3 q
        workspaceSize = (workspaceSize + static_cast<size_t>(qSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        quantFagTilingData_->dk_work_space_offset = workspaceSize;
        // matmal3 k
        workspaceSize = (workspaceSize + static_cast<size_t>(kSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        quantFagTilingData_->dv_work_space_offset = workspaceSize;
        // matmal3 v
        workspaceSize = (workspaceSize + static_cast<size_t>(vSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }
    // fp8 vScaleDs
    if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
        fBaseParams.queryType == ge::DT_HIFLOAT8) {
        workspaceSize = (workspaceSize + fBaseParams.coreNum * ALIGN128 * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }

    // mask bool workspace size
    if (fBaseParams.dropoutIsDivisibleBy8 == 0) {
        workspaceSize =
            (workspaceSize + static_cast<size_t>(fBaseParams.dropMaskSize) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }

    if (fBaseParams.enablePreSfmg) {
        // softmax grad workspace size
        quantFagTilingData_->sfmg_work_space_offset = workspaceSize;
        uint64_t sfmgSize = ((fBaseParams.b * fBaseParams.n2 * fBaseParams.g - 1) * fBaseParams.s1 +
                             AlignTo(fBaseParams.s1, ALIGN128)) *
                            BIT_NUMS;
        workspaceSize = (workspaceSize + static_cast<size_t>(sfmgSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }

    GetWorkspaceSize4Deter(workspaceSize);

    workspaceSize += WORKSPACE_BUFFER;
    workspaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttentionScoreGradTilingNormalRegbase::GetWorkspaceSize4Deter(size_t &workspaceSize)
{
    if (fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_OLD)) {
        workspaceSize += (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT + NUM_TWO * fBaseParams.s2Inner) *
                         fBaseParams.sfmgdInner * fBaseParams.aicNum * FP32_BYTES * NUM_TWO;
        // NUM_THREE: querGmOffset, keyGmOffset and valueGmOffset
        workspaceSize += fBaseParams.maxValidBBLen * fBaseParams.aicNum * INT64_BLOCK_NUM * NUM_THREE * INT64_BYTES;
    }
}

uint64_t QuantFlashAttentionScoreGradTilingNormalRegbase::GetTilingKey() const
{
    uint32_t has_attn_mask = fBaseParams.hasAttnMask ? 1 : 0;
    uint32_t has_sink = fBaseParams.hasSink ? 1 : 0;
    uint32_t s1_template_num = 512;
    uint32_t s2_template_num = 512;
    uint32_t d_template_num = 128;
    uint32_t is_n_equal = 1;
    uint32_t layout = fBaseParams.layoutType;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint8_t>(has_attn_mask), static_cast<uint8_t>(has_sink), static_cast<uint16_t>(s1_template_num),
        static_cast<uint16_t>(s2_template_num), static_cast<uint8_t>(d_template_num), static_cast<uint8_t>(is_n_equal),
        static_cast<uint8_t>(layout));
    OP_LOGI(context_, "QuantFAGTiling DoTiling success, tiling is %lu.", tilingKey);
    return tilingKey;
}

std::tuple<uint32_t, uint32_t, uint32_t> QuantFlashAttentionScoreGradTilingNormalRegbase::FuzzyForBestSplit()
{
    auto s1s2TemplateSize = GetS1S2TemplateType(fBaseParams);
    uint32_t s1Inner = s1s2TemplateSize.first / 2;
    uint32_t s2Inner = s1s2TemplateSize.second;
    uint32_t dInner = GetDTemplateType(fBaseParams);
    return std::tie(s1Inner, s2Inner, dInner);
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::PostTiling()
{
    SaveToTilingData();
    auto numBlocks = 0;
    if (fBaseParams.isDeterministic ||
        (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
         fBaseParams.queryType == ge::DT_HIFLOAT8)) {
        numBlocks = fBaseParams.aicNum;
    }
    OP_CHECK_IF(numBlocks == 0,
                OP_LOGE("QuantFlashAttentionScoreGradTilingNormalRegbase",
                        "numBlocks is 0, aicNum is %lu, aivNum is %lu.", fBaseParams.aicNum, fBaseParams.coreNum),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(numBlocks);

    // 使用SyncAll，需要设置为batch mode模式，所有核同时启动，否则在多流方式下执行可能会卡死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttentionScoreGradTilingNormalRegbase::GetParseS1S2OuterInfo(int64_t (*parseInfo)[ARRAY_LENGTH])
{
    std::vector<bool> invalidS1Array(fBaseParams.s1Outer, false);
    for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
        int64_t leftIntersectionPoint = std::max(0L, int64_t(fBaseParams.cvS2Inner * i) - fBaseParams.s2Token);
        if (leftIntersectionPoint > int64_t(fBaseParams.s1)) {
            parseInfo[i][BEGIN_IDX] = (fBaseParams.s1 + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
        } else {
            parseInfo[i][BEGIN_IDX] = leftIntersectionPoint / fBaseParams.s1CvInner;
        }
        int64_t cvBlockTail = i == fBaseParams.s2Outer - 1 ? fBaseParams.s2CvTail : fBaseParams.cvS2Inner;
        parseInfo[i][END_IDX] =
            int64_t(std::min(std::max(0L, int64_t(fBaseParams.cvS2Inner * i + cvBlockTail) + fBaseParams.s1Token),
                             int64_t(fBaseParams.s1)) +
                    fBaseParams.s1CvInner - 1) /

            fBaseParams.s1CvInner;
        int64_t tmpSize =
            (parseInfo[i][END_IDX] > parseInfo[i][BEGIN_IDX]) ? parseInfo[i][END_IDX] - parseInfo[i][BEGIN_IDX] : 0;
        if (i == 0) {
            parseInfo[i][LENGTH_IDX] = tmpSize;
        } else {
            parseInfo[i][LENGTH_IDX] = parseInfo[i - 1][LENGTH_IDX] + tmpSize;
        }
        // check invalid row or col block for BN2
        for (int64_t j = 0; j < static_cast<int64_t>(invalidS1Array.size()); j++) {
            if (j >= parseInfo[i][BEGIN_IDX] && j < parseInfo[i][END_IDX]) {
                invalidS1Array[j] = true;
            }
        }
        OP_LOGD("Sparse", " idx = %ld: Begin = %ld, End = %ld, Length = %ld, total_Length = %ld", i, parseInfo[i][0],
                parseInfo[i][1], tmpSize, parseInfo[i][LENGTH_IDX]);
    }
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::GetSparseBlockInfo()
{
    // [s2OuterIdx][begin, end, length]
    int64_t(*parseInfo)[ARRAY_LENGTH] = new int64_t[fBaseParams.s2Outer][ARRAY_LENGTH];
    GetParseS1S2OuterInfo(parseInfo);
    int64_t s1s2oCount = parseInfo[fBaseParams.s2Outer - 1][LENGTH_IDX];

    // block split
    int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * s1s2oCount;
    int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;
    int64_t blockTailTmp = fusedOuter % blockFactor;
    int64_t blockTail = blockTailTmp == 0 ? blockFactor : blockTailTmp;
    OP_LOGD("Sparse", "Sparse parseInfo fusedOuter = %ld: blockFactor = %ld, blockTail = %ld", fusedOuter, blockFactor,
            blockTail);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;
    fBaseParams.maxValidBBLen = fBaseParams.blockFactor;

    int64_t bIdx = 0;
    int64_t bTail = 0;
    int64_t n2Idx = 0;
    int64_t n2Tail = 0;
    int64_t gIdx = 0;
    int64_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s2oIdx = 0;

    int64_t n2gs1s2o = fBaseParams.n2 * fBaseParams.g * s1s2oCount;
    int64_t gs1s2o = fBaseParams.g * s1s2oCount;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] =
        fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;
    for (int64_t c = 1; c < blockOuter; c++) {
        // cal indx for total bngs1os2o(sparse)
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        bIdx = currentIdx / n2gs1s2o;
        bTail = currentIdx % n2gs1s2o;
        n2Idx = bTail / gs1s2o;
        n2Tail = bTail % gs1s2o;
        gIdx = n2Tail / s1s2oCount;
        gTail = n2Tail % s1s2oCount;

        OP_LOGD("Sparse",
                "Sparse parseInfo currentIdx = %ld: bIdx = %ld, bTail = %ld, n2Idx = %ld, n2Tail = %ld, gIdx = %ld, "
                "gTail = %ld",
                currentIdx, bIdx, bTail, n2Idx, n2Tail, gIdx, gTail);
        GetCommonS1S2OuterIndex(fBaseParams, parseInfo, gTail, s1oIdx, s2oIdx);

        // total indx in bngs1os2o (range is [))
        blockStarts[c] = (((bIdx * fBaseParams.n2 + n2Idx) * fBaseParams.g + gIdx) * fBaseParams.s2Outer + s2oIdx) *
                             fBaseParams.s1Outer +
                         s1oIdx + 1;
        blockEnds[c - 1] = blockStarts[c];
        OP_LOGD("Sparse", "blockStarts[c] = %ld:", blockStarts[c]);
    }
    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    // free tensor
    delete[] parseInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::GetSparsePrefixBlockInfo()
{
    std::vector<std::vector<std::pair<int64_t, int64_t>>> s1ValidIdx(
        fBaseParams.b, std::vector<std::pair<int64_t, int64_t>>(fBaseParams.s2Outer, {0, 0}));
    uint64_t totalValidBaseBlock = 0; // include nRation, baseN * nRation
    int32_t comBIdx = -1;
    for (int64_t bIdx = 0; bIdx < fBaseParams.b; ++bIdx) {
        int64_t prefixN = fBaseParams.prefixN[bIdx];
        if (CheckPrefixNExist(fBaseParams, bIdx, prefixN, s1ValidIdx)) {
            totalValidBaseBlock += s1ValidIdx[bIdx][fBaseParams.s2Outer - 1].second;
            continue;
        }

        if (fBaseParams.s1 <= fBaseParams.s2 - prefixN) {
            if (comBIdx != -1) {
                s1ValidIdx[bIdx].assign(s1ValidIdx[comBIdx].begin(), s1ValidIdx[comBIdx].end());
                totalValidBaseBlock += s1ValidIdx[bIdx][fBaseParams.s2Outer - 1].second;
                continue;
            }
            comBIdx = bIdx;
        }

        GetCommS1S2OuterInfo(fBaseParams, prefixN, s1ValidIdx[bIdx]);
        totalValidBaseBlock += s1ValidIdx[bIdx][fBaseParams.s2Outer - 1].second;
    }

    totalValidBaseBlock *= fBaseParams.n2 * fBaseParams.g;
    int64_t blockFactor =
        (totalValidBaseBlock + fBaseParams.aicNum - 1) / fBaseParams.aicNum; // 每个核处理的最多数据个数
    int64_t blockOuter = (static_cast<int64_t>(totalValidBaseBlock) + blockFactor - 1) / blockFactor; // 实际使用的核数

    OP_LOGD("Sparse", "Sparse parseInfo totalValidBaseBlock = %lu: blockFactor = %ld, blockOuter = %ld",
            totalValidBaseBlock, blockFactor, blockOuter);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;
    fBaseParams.maxValidBBLen = blockFactor;
    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] =
        fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;

    uint32_t coreNum = 0;
    int64_t tmepBlock = 0;
    for (int64_t bIdx = 0; bIdx < fBaseParams.b; ++bIdx) {
        for (int64_t nIdx = 0; nIdx < fBaseParams.n2; ++nIdx) {
            SetSparsePrefixBlockInterval(fBaseParams, bIdx, nIdx, s1ValidIdx, blockStarts, blockEnds, coreNum,
                                         tmepBlock);
        }
    }

    for (uint32_t coreIdx = static_cast<uint32_t>(blockOuter); coreIdx < CORE_LIST_NUM; ++coreIdx) {
        blockStarts[coreIdx] = 0;
        blockEnds[coreIdx] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::InitTilingData()
{
    QuantFlashAttnGradTiling *tilingData = this->context_->GetTilingData<QuantFlashAttnGradTiling>();
    quantFagTilingData_ = tilingData;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttentionScoreGradTilingNormalRegbase::SaveToTilingData()
{
    // set tilingdata baseinfo
    quantFagTilingData_->b = fBaseParams.b - fBaseParams.tailZeroCount;
    quantFagTilingData_->n2 = fBaseParams.n2;
    quantFagTilingData_->g = fBaseParams.g;
    quantFagTilingData_->s1 = fBaseParams.s1;
    quantFagTilingData_->d = fBaseParams.d;
    quantFagTilingData_->s2 = fBaseParams.s2;
    quantFagTilingData_->n1 = fBaseParams.n1;
    quantFagTilingData_->t1 = fBaseParams.t1;
    quantFagTilingData_->t2 = fBaseParams.t2;
    quantFagTilingData_->s1_outer = fBaseParams.s1Outer;
    quantFagTilingData_->s2_outer = fBaseParams.s2Outer;
    quantFagTilingData_->s1_tail = fBaseParams.s1CvTail;
    quantFagTilingData_->s2_tail = fBaseParams.s2CvTail;
    quantFagTilingData_->softmax_scale = fBaseParams.scaleValue;
    quantFagTilingData_->has_seq_used_q = fBaseParams.hasSequsedQ;
    quantFagTilingData_->has_seq_used_k = fBaseParams.hasSequsedKV;
    quantFagTilingData_->metadata_len = fBaseParams.metadataLen;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantFlashAttnGrad, QuantFlashAttentionScoreGradTilingNormalRegbase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 950);
} // namespace QuantFag
} // namespace optiling
