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
 * \file flash_attention_score_grad_tiling_normal_regbase.cpp
 * \brief
 */
#include "flash_attention_score_grad_tiling_normal_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "err/ops_err.h"

using namespace Ops::Transformer::OpTiling;

using namespace optiling::fag;
namespace optiling {
namespace fag {

namespace {
struct DeterBandScheduleResult {
    DeterBandScheduleMode mode = DeterBandScheduleMode::DISABLED;
    int64_t maxRound = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
};

struct BlockScheduleResult {
    bool isSplitByBlockIdx = false;
    DeterBandScheduleResult deterBandSchedule;
};

struct DeterBandScheduleParams {
    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
};

struct GQADenseScheduleResult {
    int64_t baseRound = 0;
    int64_t selectedRound = 0;
    int64_t basePeriod = 0;
    int64_t selectedPeriod = 0;
};

int64_t PositiveCeilDiv(int64_t dividend, int64_t divisor)
{
    return dividend / divisor + static_cast<int64_t>(dividend % divisor != 0);
}

int64_t PositiveGcd(int64_t lhs, int64_t rhs)
{
    while (rhs > 0) {
        const int64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

bool IsRoundGrowthWithinLimit(int64_t candidateRound, int64_t legacyRound, int64_t maxGrowthPercent,
                              int64_t percentBase)
{
    return candidateRound > 0 && legacyRound > 0 &&
           candidateRound * percentBase <= legacyRound * (percentBase + maxGrowthPercent);
}

GQADenseScheduleResult SelectGQADenseSchedule(int64_t k, int64_t m, int64_t n, int64_t b, int64_t g)
{
    GQADenseScheduleResult result;
    if (k <= 0 || m <= 0 || n <= 0 || b <= 0 || g <= 1) {
        return result;
    }

    k = std::min({k, b * g * m, b * n});
    result.baseRound = std::max({PositiveCeilDiv(b * n * g, k), PositiveCeilDiv(n, m), g});
    result.selectedRound = result.baseRound;
    const int64_t batchHeadNum = b * g;
    result.basePeriod = batchHeadNum / PositiveGcd(batchHeadNum, result.baseRound);
    result.selectedPeriod = result.basePeriod;
    if (result.baseRound % g == 0) {
        return result;
    }

    const int64_t candidateRound = PositiveCeilDiv(result.baseRound, g) * g;
    const int64_t candidatePeriod = batchHeadNum / PositiveGcd(batchHeadNum, candidateRound);
    const int64_t totalId = b * n * g;
    const int64_t candidateInvalid = k * candidateRound - totalId;
    // Aligning R to g can shorten the batch/head traversal period. Only accept it when the extra
    // rounds and invalid IDs are bounded and m has enough row offsets for the new period.
    const bool roundCostOk = (candidateRound - result.baseRound) * GQA_DENSE_PERCENT_BASE <=
                             result.baseRound * GQA_DENSE_MAX_ROUND_GROWTH_PERCENT;
    const bool invalidCostOk =
        candidateInvalid * GQA_DENSE_PERCENT_BASE <= k * candidateRound * GQA_DENSE_MAX_INVALID_PERCENT;
    const bool localityBetter = candidatePeriod < result.basePeriod && candidatePeriod < k;
    const bool rowOffsetEnough = PositiveCeilDiv(k, candidatePeriod) <= m;
    if (roundCostOk && invalidCostOk && localityBetter && rowOffsetEnough) {
        result.selectedRound = candidateRound;
        result.selectedPeriod = candidatePeriod;
    }
    return result;
}

bool IsTndDeterSwizzleScheduleSafe(const FuzzyBaseInfoParamsRegbase &params)
{
    if (params.aicNum == 0 || params.n1 <= 0) {
        return false;
    }

    const int64_t cubeBaseM = static_cast<int64_t>(params.s1Inner) * params.s1CvRatio;
    const int64_t cubeBaseN = static_cast<int64_t>(params.s2Inner) * params.s2CvRatio;
    if (cubeBaseM <= 0 || cubeBaseN <= 0 || params.actualSeqQlen.size() != params.actualSeqKvlen.size()) {
        return false;
    }

    // Mode 2 assigns consecutive linear IDs to S2 blocks inside each head and rotates S1 by S2 + round.
    // One head has at most min(k, n) active cores in a round. The same number of S1 blocks is required to
    // keep their dQ coordinates unique. The Python recommendation m >= n is the k >= n special case.
    for (size_t bIdx = 0; bIdx < params.actualSeqQlen.size(); ++bIdx) {
        const int64_t actualSeqQLen = params.actualSeqQlen[bIdx];
        const int64_t actualSeqKvLen = params.actualSeqKvlen[bIdx];
        if (actualSeqQLen <= 0 || actualSeqKvLen <= 0) {
            return false;
        }
        const int64_t m = PositiveCeilDiv(actualSeqQLen, cubeBaseM);
        const int64_t n = PositiveCeilDiv(actualSeqKvLen, cubeBaseN);
        if (m < std::min(static_cast<int64_t>(params.aicNum), n)) {
            return false;
        }
    }
    return true;
}

bool IsTndDeterSwizzlePerformanceEnough(const FuzzyBaseInfoParamsRegbase &params)
{
    if (params.aicNum == 0 || params.n2 <= 0 || params.g <= 0 ||
        params.actualSeqQlen.size() != params.actualSeqKvlen.size()) {
        return false;
    }

    const int64_t aicNum = static_cast<int64_t>(params.aicNum);
    const int64_t cubeBaseM = static_cast<int64_t>(params.s1Inner) * params.s1CvRatio;
    const int64_t cubeBaseN = static_cast<int64_t>(params.s2Inner) * params.s2CvRatio;
    if (cubeBaseM <= 0 || cubeBaseN <= 0) {
        return false;
    }

    int64_t validSlotNum = 0;
    int64_t totalSlotNum = 0;
    int64_t maxS1Outer = 0;
    int64_t maxS2Outer = 0;
    for (size_t bIdx = 0; bIdx < params.actualSeqQlen.size(); ++bIdx) {
        const int64_t actualSeqQLen = params.actualSeqQlen[bIdx];
        const int64_t actualSeqKvLen = params.actualSeqKvlen[bIdx];
        if (actualSeqQLen <= 0 || actualSeqKvLen <= 0) {
            return false;
        }
        const int64_t m = PositiveCeilDiv(actualSeqQLen, cubeBaseM);
        const int64_t n = PositiveCeilDiv(actualSeqKvLen, cubeBaseN);
        const int64_t columnNum = n * params.n2 * params.g;
        const int64_t batchRound = PositiveCeilDiv(columnNum, aicNum) * m;
        validSlotNum += m * columnNum;
        totalSlotNum += batchRound * aicNum;
        maxS1Outer = std::max(maxS1Outer, m);
        maxS2Outer = std::max(maxS2Outer, n);
    }
    const bool slotUtilizationEnough =
        totalSlotNum > 0 &&
        validSlotNum * TND_DETER_SWIZZLE_PERCENT_BASE >= totalSlotNum * TND_DETER_SWIZZLE_MIN_SLOT_UTILIZATION_PERCENT;
    const int64_t candidateMaxRound = totalSlotNum / aicNum;
    // Keep the legacy bound identical to CalcleTNDDenseDeterParam.
    const int64_t legacyMaxRound = std::max({PositiveCeilDiv(validSlotNum, aicNum), maxS1Outer * params.g, maxS2Outer});
    const bool roundGrowthWithinLimit = IsRoundGrowthWithinLimit(
        candidateMaxRound, legacyMaxRound, TND_DETER_SWIZZLE_MAX_ROUND_GROWTH_PERCENT, TND_DETER_SWIZZLE_PERCENT_BASE);
    return slotUtilizationEnough && roundGrowthWithinLimit;
}

bool IsTndDeterSwizzleSupported(const FuzzyBaseInfoParamsRegbase &params)
{
    if (!IsTndDeterSwizzleScheduleSafe(params)) {
        return false;
    }
    return IsTndDeterSwizzlePerformanceEnough(params);
}

void ConfigureTndDeterBn2S2Swizzle(FuzzyBaseInfoParamsRegbase &params, const TndBaseInfo &tndBaseInfo)
{
    // Mode 2 keeps every (B, N2, S2) column on one core, so dK/dV never need a cross-core tail merge.
    std::fill(std::begin(params.deterPrefix2), std::end(params.deterPrefix2), static_cast<int64_t>(-1));

    // Different columns still atomically accumulate dQ. Synchronize consecutive rounds to keep the accumulation
    // order deterministic, as in the BN2GS1S2 deterministic swizzle path.
    std::fill(std::begin(params.startNeedSyncRound), std::end(params.startNeedSyncRound), static_cast<uint64_t>(0));
    std::fill(std::begin(params.endNeedSyncRound), std::end(params.endNeedSyncRound), static_cast<uint64_t>(0));
    params.startNeedSyncRound[0] = 1;
    params.endNeedSyncRound[0] = tndBaseInfo.tndS2BlockPrefixSum[params.b];
}

int64_t CalcCausalSingleBatchRound(int64_t k, int64_t causalSize)
{
    if (k <= 0 || causalSize <= 0) {
        return 0;
    }
    // Keep this bound identical to CalCausalSingleBatchDeterIndex. Every complete 2*k-column
    // group consumes a shrinking rectangle; the remaining at most 2*k columns form one tail.
    const int64_t groupCount = causalSize / (NUM_TWO * k);
    const int64_t groupRound = (NUM_TWO * causalSize + 1) * groupCount - NUM_TWO * k * groupCount * groupCount;
    const int64_t remain = causalSize - NUM_TWO * k * groupCount;
    if (remain <= k) {
        return groupRound + remain;
    }
    return groupRound + std::max(remain, NUM_TWO * remain - NUM_TWO * k + 1);
}

DeterBandScheduleParams NormalizeDeterBandScheduleParams(int64_t m, int64_t n, int64_t p, int64_t q)
{
    // Keep the clamp and shifted-BAND conversion identical to CalDeterMaxLoopNum on the kernel side.
    p = std::min(p, m);
    q = std::min(q, n);
    if (p < 0) {
        return {m, n + p, 1, p + q};
    }
    if (q < 0) {
        return {m + q, n, p + q, 1};
    }
    return {m, n, p, q};
}

int64_t CalcLegacyDeterBandMaxRound(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                    int64_t hostMaxRound)
{
    // Keep rm2 identical to GenBandInfo and CalDeterMaxLoopNum on the kernel side.
    const int64_t rm2 =
        p + q > m ? m * PositiveCeilDiv(n * b, std::min(k, b * m)) : PositiveCeilDiv(n * b, k) * (p + q - 1);
    return std::max(hostMaxRound, rm2);
}

DeterBandScheduleResult SelectDeterBandSchedule(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                                int64_t hostMaxRound)
{
    DeterBandScheduleResult result;
    if (k <= 0 || m <= 0 || n <= 0 || p <= 0 || q <= 0 || b <= 0) {
        return result;
    }

    // Keep this preprocessing identical to GenBandHybridInfo on the kernel side.
    p = std::min(p, m);
    q = std::min(q, n);
    const int64_t legacyMaxRound = CalcLegacyDeterBandMaxRound(k, m, n, p, q, b, hostMaxRound);
    m = std::min(m, n + p - 1);
    n = std::min(n, m + q - 1);

    int64_t l1 = 0;
    int64_t l2 = 0;
    int64_t l3 = 0;
    int64_t bandBlocks = 0;
    if (p + q <= m) {
        l1 = q - 1;
        l2 = std::min(n - q + 1, m + NUM_TWO - p - q);
        l3 = std::max<int64_t>(0, std::min(p + n - m - 1, p + q - NUM_TWO));
        bandBlocks = (NUM_TWO * p - NUM_TWO + q) * l1 / NUM_TWO + (p + q - 1) * l2 + (p + q - NUM_TWO) * l3 -
                     l3 * (l3 - 1) / NUM_TWO;
    } else {
        l1 = m - p;
        l2 = p + q - m;
        l3 = std::min(n - q, m - 1);
        bandBlocks = (p + m - 1) * l1 / NUM_TWO + m * l2 + (NUM_TWO * m - 1 - l3) * l3 / NUM_TWO;
    }

    const int64_t pairCount = std::max<int64_t>(0, std::min(l1, l3 - p + 1));
    const int64_t colsPerBatch = l1 + l2 + l3 - pairCount;
    const int64_t bandSlot = p + q - 1;
    result.mode = DeterBandScheduleMode::BAND;
    result.maxRound = PositiveCeilDiv(b * colsPerBatch, k) * bandSlot;
    result.m = m;
    result.n = n;
    result.p = p;
    result.q = q;

    // Dense is a safe superset only when all k cores remain active and the columns issued to one
    // batch in a round cannot wrap by m and hit the same row. This is less restrictive than m >= k.
    const int64_t denseK = std::min(k, m * b);
    if (denseK == k && std::min(denseK, n) <= m) {
        const int64_t denseRound = PositiveCeilDiv(n * b, denseK) * m;
        if (denseRound < result.maxRound) {
            result.mode = DeterBandScheduleMode::DENSE;
            result.maxRound = denseRound;
        }
    }

    // Embed a near-lower BAND into a square causal triangle.  The kernel maps x back by q - 1;
    // IsValidForDeter removes the padding and the part truncated by p.  Upper embedding is not used
    // because transposing Mode03 would break the fixed-core-per-original-column requirement.
    const int64_t lowerCausalSize = m + q - 1;
    const int64_t upperCausalSize = n + p - 1;
    const int64_t lowerWaste = lowerCausalSize * (lowerCausalSize + 1) / NUM_TWO - bandBlocks;
    const int64_t upperWaste = upperCausalSize * (upperCausalSize + 1) / NUM_TWO - bandBlocks;
    bool useLowerCausal =
        bandBlocks > 0 && lowerWaste >= 0 && lowerWaste <= upperWaste && lowerWaste <= (bandBlocks - 1) / 10;
    if (useLowerCausal) {
        const int64_t causalPairCount = b / NUM_TWO;
        const int64_t causalK = std::min(k, lowerCausalSize * causalPairCount);
        // CalCausalSwizzleIndex uses the dense swizzle internally, so do not select it when that
        // helper would reduce the active-core count below k.
        if (causalPairCount > 0 && causalK == k && causalK <= lowerCausalSize) {
            const int64_t pairRound =
                PositiveCeilDiv((lowerCausalSize + 1) * causalPairCount, causalK) * lowerCausalSize;
            const int64_t tailRound = (b & 1) == 0 ? 0 : CalcCausalSingleBatchRound(k, lowerCausalSize);
            const int64_t causalRound = pairRound + tailRound;
            if (causalRound < result.maxRound) {
                result.mode = DeterBandScheduleMode::CAUSAL;
                result.maxRound = causalRound;
            }
        }
    }

    // All hybrid schedules iterate k slots in each round. Reject low-utilization schedules because
    // invalid coordinates still incur round traversal and index-calculation overhead in the kernel.
    const int64_t validSlotNum = bandBlocks * b;
    const int64_t totalSlotNum = k * result.maxRound;
    const bool slotUtilizationEnough =
        totalSlotNum > 0 && validSlotNum * DETER_BAND_SWIZZLE_PERCENT_BASE >=
                                totalSlotNum * DETER_BAND_SWIZZLE_MIN_SLOT_UTILIZATION_PERCENT;
    const bool roundGrowthWithinLimit = IsRoundGrowthWithinLimit(
        result.maxRound, legacyMaxRound, DETER_BAND_SWIZZLE_MAX_ROUND_GROWTH_PERCENT, DETER_BAND_SWIZZLE_PERCENT_BASE);
    if (!slotUtilizationEnough || !roundGrowthWithinLimit) {
        return {};
    }
    return result;
}

BlockScheduleResult SelectBlockSchedule(const FuzzyBaseInfoParamsRegbase &params)
{
    BlockScheduleResult result;
    const bool canSplitByBlockIdx = params.enableSwizzle && params.layoutType != INPUT_FORMAT_TND &&
                                    params.splitAxis == SplitAxisEnum::BN2GS1S2 &&
                                    params.s1Inner * params.s1CvRatio == params.s2Inner * params.s2CvRatio &&
                                    params.sparseType != static_cast<uint8_t>(SparseType::UNSUPPORTED);
    result.isSplitByBlockIdx = canSplitByBlockIdx;
    if (!params.isDeterministic) {
        return result;
    }

    const bool causalCond = params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL);
    const bool rightDownBandCond = params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) &&
                                   params.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL);
    result.isSplitByBlockIdx = canSplitByBlockIdx && (((params.b * params.n2) & 1) == 0) && params.g == 1 &&
                               params.s1 >= params.aicNum * static_cast<uint32_t>(ConstAxisTemplateNum::NUM128) &&
                               (params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_DENSE) ||
                                causalCond || rightDownBandCond);

    // Preserve the legacy RIGHT_DOWN_CAUSAL swizzle whenever its original entry conditions are met.
    // With schedule mode DISABLED, the kernel keeps using CalCausalSwizzleIndex and its matching
    // max-round formula. Re-selecting this path as a hybrid DENSE schedule can increase the rounds.
    if (rightDownBandCond && result.isSplitByBlockIdx) {
        return result;
    }

    const int64_t actualBatch = (params.b - params.tailZeroCount) * params.n2;
    // DETER_BAND is the scheduling contract here. It covers BAND, compatible NO_MASK, and
    // unequal RIGHT_DOWN_CAUSAL after their tokens have been normalized to the same band geometry.
    const bool hybridBandCond = canSplitByBlockIdx &&
                                params.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) &&
                                params.g == 1 && params.coreNum == params.aicNum * NUM_TWO && actualBatch > 0;
    if (!hybridBandCond) {
        return result;
    }

    const int64_t cubeBase = params.s1Inner * params.s1CvRatio;
    const int64_t serializedS1Token = std::min<int64_t>(params.s1Token, INT32_MAX);
    const int64_t serializedS2Token = std::min<int64_t>(params.s2Token, INT32_MAX);
    const int64_t p = CeilDivideBy(serializedS1Token, cubeBase) + 1;
    const int64_t q = CeilDivideBy(serializedS2Token, cubeBase) + 1;
    const DeterBandScheduleParams scheduleParams =
        NormalizeDeterBandScheduleParams(params.s1Outer, params.s2Outer, p, q);
    result.deterBandSchedule =
        SelectDeterBandSchedule(static_cast<int64_t>(params.aicNum), scheduleParams.m, scheduleParams.n,
                                scheduleParams.p, scheduleParams.q, actualBatch, params.deterMaxRound);
    if (result.deterBandSchedule.mode != DeterBandScheduleMode::DISABLED) {
        result.isSplitByBlockIdx = true;
    }
    return result;
}
} // namespace

bool IsSameShape(const gert::StorageShape *aShape, const gert::StorageShape *bShape)
{
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr),
                OP_LOGW("flash_attention_score_grad_tiling_normal_regbase", "aShape or bShape is nullptr."),
                return false);
    uint32_t dimSizeA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimSizeB = bShape->GetStorageShape().GetDimNum();
    if (dimSizeA != dimSizeB) {
        return false;
    }

    for (uint32_t i = 0; i < dimSizeA; i++) {
        auto dimA = aShape->GetStorageShape().GetDim(i);
        auto dimB = bShape->GetStorageShape().GetDim(i);
        if (dimA != dimB) {
            return false;
        }
    }
    return true;
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::GetShapeAttrsInfo()
{
    fBaseParams.isDeterministic = (context_->GetDeterministic() == 1);
    const gert::StorageShape *queryShape = context_->GetInputShape(QUERY_IDX); // [B, N2, G, S1, D]
    const gert::StorageShape *keyShape = context_->GetInputShape(KEY_IDX);     // [B, N2, 1, S2, D]
    const gert::StorageShape *valueShape = context_->GetInputShape(VALUE_IDX); // [B, N2, 1, S2, D_V]

    int64_t headNum = *context_->GetAttrs()->GetAttrPointer<int>(HEAD_ATTR_IDX);
    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);

    // get rope
    auto queryRope = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::QUERY_ROPE_IDX));
    const gert::Shape *queryRopeShape = &queryRope->GetStorageShape();
    bool hasQueryRope = queryRope != nullptr && queryRopeShape->GetDimNum() != 0;
    auto keyRope = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::KEY_ROPE_IDX));
    const gert::Shape *keyRopeShape = &keyRope->GetStorageShape();

    // get dy/attentionIn
    auto dy = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::DY));
    const gert::Shape *dyShape = &dy->GetStorageShape();
    auto attentionIn = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::ATTENTION_IN));
    const gert::Shape *attentionInShape = &attentionIn->GetStorageShape();
    const gert::Shape &qShape = queryShape->GetStorageShape();
    const gert::Shape &kShape = keyShape->GetStorageShape();
    const gert::Shape &vShape = valueShape->GetStorageShape();

    std::string qdyShapesMsg = "{" + Ops::Base::ToString(qShape) + ", " + Ops::Base::ToString(*dyShape) + "}";
    std::string kvShapesMsg = "{" + Ops::Base::ToString(kShape) + ", " + Ops::Base::ToString(vShape) + "}";
    std::string vdyShapesMsg = "{" + Ops::Base::ToString(vShape) + ", " + Ops::Base::ToString(*dyShape) + "}";

    if (!IsSameShape(dy, attentionIn)) {
        std::string shapesMsg =
            "{" + Ops::Base::ToString(*dyShape) + ", " + Ops::Base::ToString(*attentionInShape) + "}";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "dy, attentionInOptional", shapesMsg.c_str(),
                                               "The shapes of dy and attentionInOptional must be the same");
        return ge::GRAPH_PARAM_INVALID;
    }

    bool hasKeyRope = keyRope != nullptr && keyRopeShape->GetDimNum() != 0;
    if (hasQueryRope ^ hasKeyRope) {
        std::string qrShape = hasQueryRope ? Ops::Base::ToString(*queryRopeShape) : "null";
        std::string krShape = hasKeyRope ? Ops::Base::ToString(*keyRopeShape) : "null";
        std::string shapeMsg = "{" + qrShape + ", " + krShape + "}";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "FlashAttentionScoreGrad", "queryRopeOptional, keyRopeOptional", shapeMsg.c_str(),
            "queryRopeOptional, keyRopeOptional cannot be empty tensors, if queryRopeOptional is an "
            "empty tensor, keyRopeOptional must also be an empty tensor");
        return ge::GRAPH_PARAM_INVALID;
    }
    fBaseParams.hasRope = hasQueryRope && hasKeyRope;
    int64_t qRopeD = 0;
    int64_t kRopeD = 0;

    if (strcmp(inputLayout, "SBH") == 0) {
        OP_LOGD(context_, "inputLayout == SBH queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_S2BN2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.n2 = headNum / fBaseParams.g;
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.d =
            fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / headNum; // H=N*D
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_2) / fBaseParams.n2;                // H=N2*D1
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_0);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_2) / headNum : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_2) / fBaseParams.n2 : 0;

        std::string hAxisMsg =
            "When inputLayout is SBH, h axis of dy must be exactly divisible " + std::to_string(headNum);
        OP_CHECK_IF(dyShape->GetDim(INPUT_DIM_2) % headNum != 0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "dy",
                                                           Ops::Base::ToString(*dyShape).c_str(), hAxisMsg.c_str()),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((qShape.GetDim(INPUT_DIM_0) != dyShape->GetDim(INPUT_DIM_0)) ||
                        (qShape.GetDim(INPUT_DIM_1) != dyShape->GetDim(INPUT_DIM_1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "query, dy", qdyShapesMsg.c_str(),
                        "B axis and s axis of dy must be equal to b axis and s axis of query"),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((kShape.GetDim(INPUT_DIM_0) != vShape.GetDim(INPUT_DIM_0)) ||
                        (kShape.GetDim(INPUT_DIM_1) != vShape.GetDim(INPUT_DIM_1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "keyIn, value", kvShapesMsg.c_str(),
                        "B axis and s axis of value must be equal to b axis and s axis of keyIn"),
                    return ge::GRAPH_PARAM_INVALID);
        auto dyDDim = dyShape->GetDim(INPUT_DIM_2) / headNum;
        auto qkDDim = qShape.GetDim(INPUT_DIM_2) / headNum;
        auto vNdim = kShape.GetDim(INPUT_DIM_2) / qkDDim;
        auto vDDim = vShape.GetDim(INPUT_DIM_2) / vNdim;
        OP_CHECK_IF(vDDim != dyDDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "value, dy", vdyShapesMsg.c_str(),
                                                           "D axis of dy must be equal to d axis of value"),
                    return ge::GRAPH_PARAM_INVALID);
    } else if (strcmp(inputLayout, "BSH") == 0) {
        OP_LOGD(context_, "inputLayout == BSH queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.n2 = headNum / fBaseParams.g;
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d =
            fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / headNum; // H=N*D
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_2) / fBaseParams.n2;                // H=N2*D1
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_2) / headNum : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_2) / fBaseParams.n2 : 0;

        std::string hAxisMsg =
            "When inputLayout is BSH, h axis of dy must be exactly divisible " + std::to_string(headNum);
        OP_CHECK_IF((qShape.GetDim(INPUT_DIM_0) != dyShape->GetDim(INPUT_DIM_0)) ||
                        (qShape.GetDim(INPUT_DIM_1) != dyShape->GetDim(INPUT_DIM_1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "query, dy", qdyShapesMsg.c_str(),
                        "B axis and s axis of dy must be equal to b axis and s axis of query"),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(dyShape->GetDim(INPUT_DIM_2) % headNum != 0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "dy",
                                                           Ops::Base::ToString(*dyShape).c_str(), hAxisMsg.c_str()),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((kShape.GetDim(INPUT_DIM_0) != vShape.GetDim(INPUT_DIM_0)) ||
                        (kShape.GetDim(INPUT_DIM_1) != vShape.GetDim(INPUT_DIM_1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "keyIn, value", kvShapesMsg.c_str(),
                        "B axis and s axis of value must be equal to b axis and s axis of keyIn"),
                    return ge::GRAPH_PARAM_INVALID);
        auto qkDDim = qShape.GetDim(INPUT_DIM_2) / headNum;
        auto vNdim = kShape.GetDim(INPUT_DIM_2) / qkDDim;
        auto vDDim = vShape.GetDim(INPUT_DIM_2) / vNdim;
        auto dyDDim = dyShape->GetDim(INPUT_DIM_2) / headNum;
        OP_CHECK_IF(vDDim != dyDDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "value, dy", vdyShapesMsg.c_str(),
                                                           "D axis of dy must be equal to d axis of value"),
                    return ge::GRAPH_PARAM_INVALID);
    } else if (strcmp(inputLayout, "BNSD") == 0) {
        OP_LOGD(context_, "inputLayout == BNSD queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_BN2GS2D;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_1) / keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.d = fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_3) : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_3) : 0;
        OP_LOGD(context_, "inputLayout == BNSD queryShape", "%ld, %ld, %ld, %ld,",
                queryShape->GetStorageShape().GetDim(INPUT_DIM_0), queryShape->GetStorageShape().GetDim(INPUT_DIM_1),
                queryShape->GetStorageShape().GetDim(INPUT_DIM_2), queryShape->GetStorageShape().GetDim(INPUT_DIM_3));

        std::string hAxisMsg = "When inputLayout is BNSD, N axis of dy must be equal to " + std::to_string(headNum);
        OP_CHECK_IF(dyShape->GetDim(INPUT_DIM_1) != headNum,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "dy",
                                                           Ops::Base::ToString(*dyShape).c_str(), hAxisMsg.c_str()),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((qShape.GetDim(INPUT_DIM_0) != dyShape->GetDim(INPUT_DIM_0)) ||
                        (qShape.GetDim(INPUT_DIM_1) != dyShape->GetDim(INPUT_DIM_1)) ||
                        (qShape.GetDim(INPUT_DIM_2) != dyShape->GetDim(INPUT_DIM_2)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "query, dy", qdyShapesMsg.c_str(),
                        "B axis, s axis and n axis of dy must be equal to b axis, s axis and n axis of query"),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((kShape.GetDim(INPUT_DIM_0) != vShape.GetDim(INPUT_DIM_0)) ||
                        (kShape.GetDim(INPUT_DIM_1) != vShape.GetDim(INPUT_DIM_1)) ||
                        (kShape.GetDim(INPUT_DIM_2) != vShape.GetDim(INPUT_DIM_2)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "keyIn, value", kvShapesMsg.c_str(),
                        "B axis, s axis and n axis of value must be equal to b axis, s axis and n axis of keyIn"),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(vShape.GetDim(INPUT_DIM_3) != dyShape->GetDim(INPUT_DIM_3),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "value, dy", vdyShapesMsg.c_str(),
                                                           "D axis of dy must be equal to d axis of value"),
                    return ge::GRAPH_PARAM_INVALID);
    } else if (strcmp(inputLayout, "TND") == 0) {
        OP_LOGD(context_, "inputLayout == TND");
        fBaseParams.layoutType = INPUT_FORMAT_TND;

        auto actualSeqQlenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_Q_LEN));
        auto actualSeqKvlenTensor =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
        if (actualSeqQlenTensor == nullptr || actualSeqKvlenTensor == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "actualSeqQLenOptional, actualSeqKvLenOptional");
            return ge::GRAPH_PARAM_INVALID;
        }

        const size_t seqQShapeSize = actualSeqQlenTensor->GetShapeSize();
        const size_t kvSeqShapeSize = actualSeqKvlenTensor->GetShapeSize();
        if (seqQShapeSize != kvSeqShapeSize) {
            std::string shapeSizeMsg = std::to_string(seqQShapeSize) + ", " + std::to_string(kvSeqShapeSize);
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                "FlashAttentionScoreGrad", "actualSeqQLenOptional, actualSeqKvLenOptional", shapeSizeMsg.c_str(),
                "The shape sizes of actualSeqQLenOptional and actualSeqKvLenOptional must be same");
            return ge::GRAPH_PARAM_INVALID;
        }

        const int64_t *qValue = actualSeqQlenTensor->GetData<int64_t>();
        const int64_t *kvValue = actualSeqKvlenTensor->GetData<int64_t>();

        int64_t lastQLen = 0;
        int64_t lastKvLen = 0;
        fBaseParams.isS1S2Same = true;
        fBaseParams.isAllSame = true;
        bool isEOD = false;
        for (size_t i = 0; i < seqQShapeSize; i++) {
            if (i == static_cast<size_t>(0)) {
                fBaseParams.actualSeqQlen.push_back(qValue[i]);
                fBaseParams.actualSeqKvlen.push_back(kvValue[i]);
                if (qValue[0] == 0 || kvValue[0] == 0) {
                    fBaseParams.sValueZeroUnderTND = true;
                    tndBaseInfo.isSeqExistZero = true;
                }
                tndBaseInfo.isS1GreaterThanS2 = tndBaseInfo.isS1GreaterThanS2 && (qValue[0] >= kvValue[0]);
                tndBaseInfo.isS1LessThanS2 = tndBaseInfo.isS1LessThanS2 && (qValue[0] <= kvValue[0]);
            } else {
                lastQLen = fBaseParams.actualSeqQlen[i - 1];
                lastKvLen = fBaseParams.actualSeqKvlen[i - 1];
                auto qLen = qValue[i] - qValue[i - 1];
                auto kvLen = kvValue[i] - kvValue[i - 1];
                fBaseParams.actualSeqQlen.push_back(qLen < 0 ? 0 : qLen);
                fBaseParams.actualSeqKvlen.push_back(kvLen < 0 ? 0 : kvLen);
                if (qLen < 0 || kvLen < 0) {
                    isEOD = true;
                }
                tndBaseInfo.isSeqExistZero = (tndBaseInfo.isSeqExistZero || (qLen == 0 || kvLen == 0));
                if (isEOD && (qValue[i] == 0 || kvValue[i] == 0)) {
                    ++fBaseParams.tailZeroCount;
                    fBaseParams.sValueZeroUnderTND = true;
                } else if (isEOD && (qValue[i] != 0 || kvValue[i] != 0)) {
                    std::string valuesMsg = "{" + std::to_string(qValue[i]) + ", " + std::to_string(kvValue[i]) + "}";
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "FlashAttentionScoreGrad", "actualSeqQlenOptional, actualSeqKvlenOptional", valuesMsg.c_str(),
                        "When inputLayout is TND EOD, the values of last several actualSeqQlenOptional and "
                        "actualSeqKvlenOptional must be 0");
                    return ge::GRAPH_PARAM_INVALID;
                }
                fBaseParams.isAllSame = (kvValue[i] - kvValue[i - 1] == lastKvLen) &&
                                        (qValue[i] - qValue[i - 1] == lastQLen) && fBaseParams.isAllSame;
                tndBaseInfo.isS1GreaterThanS2 = tndBaseInfo.isS1GreaterThanS2 && (qLen >= kvLen);
                tndBaseInfo.isS1LessThanS2 = tndBaseInfo.isS1LessThanS2 && (qLen <= kvLen);
            }
            fBaseParams.isS1S2Same =
                fBaseParams.actualSeqQlen[i] == fBaseParams.actualSeqKvlen[i] && fBaseParams.isS1S2Same;
            fBaseParams.sumS1S2Product += fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i];
        }

        fBaseParams.s1 = *std::max_element(fBaseParams.actualSeqQlen.begin(), fBaseParams.actualSeqQlen.end());
        fBaseParams.s2 = *std::max_element(fBaseParams.actualSeqKvlen.begin(), fBaseParams.actualSeqKvlen.end());
        fBaseParams.t1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.t2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.b = seqQShapeSize;
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_1) / keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d = fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_2);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_2) : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_2) : 0;

        std::string hAxisMsg = "When inputLayout is TND, N axis of dy must be equal to " + std::to_string(headNum);
        OP_CHECK_IF(dyShape->GetDim(INPUT_DIM_1) != headNum,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "dy",
                                                           Ops::Base::ToString(*dyShape).c_str(), hAxisMsg.c_str()),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((qShape.GetDim(INPUT_DIM_0) != dyShape->GetDim(INPUT_DIM_0)) ||
                        (qShape.GetDim(INPUT_DIM_1) != dyShape->GetDim(INPUT_DIM_1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "query, dy", qdyShapesMsg.c_str(),
                        "T axis and n axis of dy must be equal to t axis and n axis of query"),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((kShape.GetDim(INPUT_DIM_0) != vShape.GetDim(INPUT_DIM_0)) ||
                        (kShape.GetDim(INPUT_DIM_1) != vShape.GetDim(INPUT_DIM_1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "keyIn, value", kvShapesMsg.c_str(),
                        "T axis and n axis of value must be equal to t axis and n axis of keyIn"),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(vShape.GetDim(INPUT_DIM_2) != dyShape->GetDim(INPUT_DIM_2),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "value, dy", vdyShapesMsg.c_str(),
                                                           "D axis of dy must be equal to d axis of value"),
                    return ge::GRAPH_PARAM_INVALID);
    } else {
        OP_LOGD(context_, "inputLayout == BSND queryShape");
        // inputLayout = "BSND"
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d = fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_3) : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_3) : 0;

        std::string hAxisMsg = "When inputLayout is BSND, N axis of dy must be equal to " + std::to_string(headNum);
        OP_CHECK_IF(dyShape->GetDim(INPUT_DIM_2) != headNum,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "dy",
                                                           Ops::Base::ToString(*dyShape).c_str(), hAxisMsg.c_str()),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((qShape.GetDim(INPUT_DIM_0) != dyShape->GetDim(INPUT_DIM_0)) ||
                        (qShape.GetDim(INPUT_DIM_1) != dyShape->GetDim(INPUT_DIM_1)) ||
                        (qShape.GetDim(INPUT_DIM_2) != dyShape->GetDim(INPUT_DIM_2)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "query, dy", qdyShapesMsg.c_str(),
                        "B axis, s axis and n axis of dy must be equal to b axis, s axis and n axis of query"),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF((kShape.GetDim(INPUT_DIM_0) != vShape.GetDim(INPUT_DIM_0)) ||
                        (kShape.GetDim(INPUT_DIM_1) != vShape.GetDim(INPUT_DIM_1)) ||
                        (kShape.GetDim(INPUT_DIM_2) != vShape.GetDim(INPUT_DIM_2)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "keyIn, value", kvShapesMsg.c_str(),
                        "B axis, s axis and n axis of value must be equal to b axis, s axis and n axis of keyIn"),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(vShape.GetDim(INPUT_DIM_3) != dyShape->GetDim(INPUT_DIM_3),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "value, dy", vdyShapesMsg.c_str(),
                                                           "D axis of dy must be equal to d axis of value"),
                    return ge::GRAPH_PARAM_INVALID);
    }

    // For fixed-length layouts, S1/S2 equality comes directly from the Q/K shapes.
    // TND initializes this flag from every pair of actual sequence lengths above.
    if (fBaseParams.layoutType != INPUT_FORMAT_TND) {
        fBaseParams.isS1S2Same = (fBaseParams.s1 == fBaseParams.s2);
    }

    // check rope
    if (fBaseParams.hasRope) {
        if (qRopeD != kRopeD || qRopeD != ROPE_D_64) {
            std::string shapesMsg = Ops::Base::ToString(*queryRopeShape) + ", " + Ops::Base::ToString(*keyRopeShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "queryRopeOptional, keyRopeOptional",
                                                   shapesMsg.c_str(),
                                                   "D of queryRopeOptional and keyRopeOptional must be equal to 64");
            return ge::GRAPH_PARAM_INVALID;
        }
    }

    fBaseParams.n1 = fBaseParams.n2 * fBaseParams.g;
    return ProcessOptionalInput(context_, fBaseParams);
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::GetPlatformInfo()
{
    uint32_t coreNum = CORE_INIT_NUM; // 40 is init core num

    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context_->GetCompileInfo());
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

bool FlashAttentionScoreGradTilingNormalRegbase::IsCapable()
{
    const char *tndSoftmaxIn =
        context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN) ?
            context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN)) :
            "";
    if (strcmp(tndSoftmaxIn, "") != 0)
        return false;

    // 基础模板 全部支持
    if (npuArch == NpuArch::DAV_3510) {
        OP_LOGD(context_, "FlashAttentionScoreGradTilingNormalRegbase hit");
        return true;
    }
    return false;
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::DoOpTiling()
{
    SetSplitAxis(context_, fBaseParams, tndBaseInfo);
    DoSplit();
    auto ret = DoSparse();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 分核优化，对于超出l2 cache的case优先多个核处理BN下的S1S2
    bool isExceedL2Cache = CheckExceedL2Cache();
    bool isLargeInvalidBlk = CheckIsLargeInvalidBlk(fBaseParams);
    fBaseParams.enableSwizzle = (isExceedL2Cache || isLargeInvalidBlk) && fBaseParams.blockOuter == fBaseParams.aicNum;
    // dqkv fixp nz出优化开关
    fBaseParams.isNzOut =
        (fBaseParams.splitAxis == SplitAxisEnum::BN2GS1S2 &&
         fBaseParams.d > static_cast<uint32_t>(ConstAxisTemplateNum::NUM64) &&
         fBaseParams.d < static_cast<uint32_t>(ConstAxisTemplateNum::NUM128) && fBaseParams.d % FP16_C0_SIZE != 0 &&
         !(fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
           fBaseParams.queryType == ge::DT_HIFLOAT8 || fBaseParams.queryType == ge::DT_FLOAT) &&
         fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_OLD)) &&
        fBaseParams.enableSwizzle && fBaseParams.s1 >= NZ_OUT_MIN_S_SIZE && fBaseParams.s2 >= NZ_OUT_MIN_S_SIZE;
    // TND场景下，确定性计算+MHA+Dense支持BN2GS1S2/BN2S2模板，非确定性计算支持BN2S2模板。
    // kernel侧确定性swizzle尚未适配GQA，需通过g == 1将GQA筛除。
    // Mode 2 pads every batch's (N2, S2) columns to a full AIC group. Reject schedules whose aggregate
    // valid-slot ratio is too low; otherwise short tail groups serialize batches while most cores stay idle.
    const bool deterTndSwizzleSupported = !fBaseParams.isDeterministic || IsTndDeterSwizzleSupported(fBaseParams);
    const bool templateSupportCond =
        (fBaseParams.isDeterministic &&
         (fBaseParams.splitAxis == SplitAxisEnum::BN2GS1S2 || fBaseParams.splitAxis == SplitAxisEnum::BN2S2) &&
         fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_DENSE) && fBaseParams.g == 1 &&
         deterTndSwizzleSupported) ||
        (!fBaseParams.isDeterministic && fBaseParams.splitAxis == SplitAxisEnum::BN2S2 &&
         (fBaseParams.s1 >= TND_SWIZZLE_MIN_S1_SIZE ||
          (fBaseParams.s2 > static_cast<uint32_t>(ConstAxisTemplateNum::NUM128) &&
           fBaseParams.s1 >= TND_SWIZZLE_MIN_S1_SIZE_1)) &&
         (fBaseParams.sparseType != static_cast<uint8_t>(SparseType::UNSUPPORTED)));
    tndBaseInfo.isTndSwizzle = fBaseParams.enableSwizzle && fBaseParams.layoutType == INPUT_FORMAT_TND &&
                               templateSupportCond && fBaseParams.b < TND_SWIZZLE_PREFIX_NUM &&
                               !tndBaseInfo.isSeqExistZero && fBaseParams.tailZeroCount == 0;
    if (fBaseParams.isDeterministic && fBaseParams.splitAxis == SplitAxisEnum::BN2S2 && tndBaseInfo.isTndSwizzle) {
        ConfigureTndDeterBn2S2Swizzle(fBaseParams, tndBaseInfo);
    }
    OP_LOGI(context_,
            "isExceedL2Cache=[%d], sparseType=[%d], enableSwizzle=[%d], "
            "deterTndSwizzleSupported=[%d], isTndSwizzle=[%d], isNzOut=[%d].",
            static_cast<int>(isExceedL2Cache), static_cast<int>(fBaseParams.sparseType),
            static_cast<int>(fBaseParams.enableSwizzle), static_cast<int>(deterTndSwizzleSupported),
            tndBaseInfo.isTndSwizzle, fBaseParams.isNzOut);

    ret = InitTilingData();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    DoPreTiling();
    DoPostTiling();
    DetermineMode(fBaseParams);
    DetermineBlockSchedule();
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingNormalRegbase::DetermineBlockSchedule()
{
    const BlockScheduleResult result = SelectBlockSchedule(fBaseParams);
    fBaseParams.isSplitByBlockIdx = result.isSplitByBlockIdx;
    fBaseParams.deterBandScheduleMode = result.deterBandSchedule.mode;
    if (result.deterBandSchedule.mode != DeterBandScheduleMode::DISABLED) {
        fBaseParams.deterMaxRound = result.deterBandSchedule.maxRound;
        const int64_t actualBatch = (fBaseParams.b - fBaseParams.tailZeroCount) * fBaseParams.n2;
        OP_LOGI(context_,
                "Select deterministic BAND schedule, mode=[%u], maxRound=[%ld], k=[%lu], b=[%ld], "
                "effective(m,n,p,q)=[%ld,%ld,%ld,%ld]",
                static_cast<uint32_t>(result.deterBandSchedule.mode), result.deterBandSchedule.maxRound,
                fBaseParams.aicNum, actualBatch, result.deterBandSchedule.m, result.deterBandSchedule.n,
                result.deterBandSchedule.p, result.deterBandSchedule.q);
    }
    OP_LOGI(context_, "Determine block schedule, isSplitByBlockIdx=[%d], deterBandScheduleMode=[%u]",
            static_cast<int>(fBaseParams.isSplitByBlockIdx), static_cast<uint32_t>(fBaseParams.deterBandScheduleMode));
}

void FlashAttentionScoreGradTilingNormalRegbase::DoSplit()
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
    fBaseParams.s1Tail = s1TailTmp == 0 ? s1Inner : s1TailTmp;
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
    fBaseParams.s2Tail = s2TailTmp == 0 ? s2Inner : s2TailTmp;
    fBaseParams.s2CvTail = s2CvTailTmp == 0 ? cvS2Inner : s2CvTailTmp;
    fBaseParams.s2Outer = s2Outer;
    fBaseParams.cvS2Inner = cvS2Inner;
    fBaseParams.s2Inner = s2Inner;

    uint32_t sfmgdInner = std::get<2>(bestSplitRes);
    fBaseParams.sfmgdInner = sfmgdInner;
}

bool FlashAttentionScoreGradTilingNormalRegbase::DoBn2s2Sparse()
{
    if (fBaseParams.splitAxis != SplitAxisEnum::BN2S2 ||
        fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_OLD)) {
        return false;
    }
    if (fBaseParams.isSparse || fBaseParams.layoutType == INPUT_FORMAT_TND) {
        return GetBlockInfoOfBNS4TND();
    } else {
        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];
        // block split
        int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s2Outer;
        int64_t bns2Factor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
        int64_t blockOuter = (fusedOuter + bns2Factor - 1) / bns2Factor;
        int64_t totalBlock = fusedOuter * fBaseParams.s1Outer;
        int64_t blockFactor = bns2Factor * fBaseParams.s1Outer;

        for (int64_t i = 0; i < blockOuter; i++) {
            blockStarts[i] = blockFactor * i;
            blockEnds[i] = std::min(blockFactor * (i + 1), totalBlock);
        }
        for (uint32_t i = static_cast<uint32_t>(blockOuter); i < CORE_LIST_NUM; i++) {
            blockStarts[i] = 0;
            blockEnds[i] = 0;
        }

        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

        fBaseParams.blockOuter = blockOuter;
        fBaseParams.blockFactor = blockFactor;
        fBaseParams.maxValidBBLen = blockFactor;
        return true;
    }
    return false;
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::GetSparseBlockInfoBn2()
{
    // [s2OuterIdx][begin, end, length]
    int64_t(*parseInfo)[ARRAY_LENGTH] = new int64_t[fBaseParams.s2Outer][ARRAY_LENGTH];
    GetParseS1S2OuterInfo(parseInfo);
    int64_t s1s2oCount = parseInfo[fBaseParams.s2Outer - 1][LENGTH_IDX];

    // block split
    int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g;
    int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;
    int64_t bnTail = fusedOuter % blockFactor;
    OP_LOGD("Sparse", "GetSparseBlockInfoBn2, bnNum = %ld: bnFactor = %ld, bnTail = %ld", fusedOuter, blockFactor,
            bnTail);

    fusedOuter *= s1s2oCount;
    blockFactor *= s1s2oCount;
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
                "GetSparseBlockInfoBn2, currentIdx = %ld: bIdx = %ld, bTail = %ld, n2Idx = %ld, n2Tail = %ld, gIdx = "
                "%ld, gTail = %ld",
                currentIdx, bIdx, bTail, n2Idx, n2Tail, gIdx, gTail);
        GetCommonS1S2OuterIndex(fBaseParams, parseInfo, gTail, s1oIdx, s2oIdx);

        // total indx in bngs1os2o (range is [))
        blockStarts[c] = (((bIdx * fBaseParams.n2 + n2Idx) * fBaseParams.g + gIdx) * fBaseParams.s2Outer + s2oIdx) *
                             fBaseParams.s1Outer +
                         s1oIdx + 1;
        blockEnds[c - 1] = blockStarts[c];
        OP_LOGD("Sparse", "GetSparseBlockInfoBn2, blockStarts[%ld] = %ld:", c, blockStarts[c]);
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

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::DoBn2MultiBlkSparse()
{
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        return GetBlockInfoOfTNDForBn2();
    } else if (fBaseParams.isSparse) {
        return GetSparseBlockInfoBn2();
    } else {
        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];
        // block split, Core partitioning by BN
        int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g;
        int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
        int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;
        blockFactor *= (fBaseParams.s1Outer * fBaseParams.s2Outer);
        fusedOuter *= (fBaseParams.s1Outer * fBaseParams.s2Outer);

        fBaseParams.blockOuter = blockOuter;
        fBaseParams.blockFactor = blockFactor;
        fBaseParams.maxValidBBLen = blockFactor;

        for (int64_t i = 0; i < blockOuter; i++) {
            blockStarts[i] = blockFactor * i;
            blockEnds[i] = std::min(blockFactor * (i + 1), fusedOuter);
            OP_LOGD("DoBn2MultiBlkSparse", "Normally partition, blockStarts[%ld] = %ld, blockEnds[%ld] = %ld", i,
                    blockStarts[i], i, blockEnds[i]);
        }
        for (uint32_t i = static_cast<uint32_t>(blockOuter); i < CORE_LIST_NUM; i++) {
            blockStarts[i] = 0;
            blockEnds[i] = 0;
        }

        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::DoSparse()
{
    fBaseParams.sparseType = GetSparseType(); // 非确定性计算下获取sparseType
    fBaseParams.deterSparseType = GetDeterSparseTilingKey();
    CalcleDeterParam();
    if (DoBn2s2Sparse() && fBaseParams.blockOuter >= fBaseParams.aicNum) {
        return ge::GRAPH_SUCCESS;
    } else {
        // TND S1 S2全等场景下if分支尝试走BN2S2分核优化,如果判断不能走则恢复layoutType赋值
        if (SupportTrans2BS2N2GD(fBaseParams)) {
            fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
            fBaseParams.sparseType = GetSparseType(); // 确保sparseType正确
        }
    }
    if (fBaseParams.splitAxis == SplitAxisEnum::BN2 && fBaseParams.isBn2MultiBlk) {
        bool earlyReturn = true;
        bool res = DoBn2MultiBlkSparse();
        // 当BN2多基本块场景，上方函数判断遇到无效行、列后，需要走S1S2模板，性能达到最优
        OP_LOGD("DoBn2MultiBlkSparse", "fBaseParams.isInvalidCol %d, fBaseParams.isInvalidRow %d",
                fBaseParams.isInvalidCol, fBaseParams.isInvalidRow);
        if ((fBaseParams.isInvalidCol || fBaseParams.isInvalidRow)) {
            fBaseParams.isBn2 = false;
            fBaseParams.isBn2MultiBlk = false;
            fBaseParams.isDeterministic = (context_->GetDeterministic() == 1);
            fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
            earlyReturn = false;
        }
        if (earlyReturn) {
            return res;
        }
    }
    fBaseParams.splitAxis = fBaseParams.isBn2 ? SplitAxisEnum::BN2 : SplitAxisEnum::BN2GS1S2;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        GetSparseUnpadBlockInfo();
    } else if (fBaseParams.isSparse) {
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

bool FlashAttentionScoreGradTilingNormalRegbase::CheckSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft,
                                                                         int64_t s2IdxRight, int64_t bIdx,
                                                                         int64_t blockIdx)
{
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        return CheckUnpadSparseLeftAndRight(s1oDimIdx, s2IdxLeft, s2IdxRight, bIdx);
    } else {
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
                int64_t curBIdx =
                    blockIdx / (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
                s2EndLen = std::min(std::max(s2EndLen, static_cast<int64_t>(fBaseParams.prefixN[curBIdx])),
                                    static_cast<int64_t>(fBaseParams.s2));
            }
            bool isValid = s2IdxLeft < s2EndLen;
            return isValid;
        } else {
            int64_t s2SparseLeft = std::max(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * s1oDimIdx - fBaseParams.s1Token,
                                            static_cast<int64_t>(0));
            s2SparseLeft = AlignTo(s2SparseLeft, ALIGN64);
            int64_t s2SparseRight =
                AlignTo(std::min(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), fBaseParams.s1) +
                            fBaseParams.s2Token,
                        ALIGN64);
            s2SparseRight = std::min(s2SparseRight, fBaseParams.s2);
            bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
            return isValid;
        }
    }
}

bool FlashAttentionScoreGradTilingNormalRegbase::IsValid(int64_t blockIdx)
{
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        return IsValidUnpad(blockIdx);
    } else {
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
}

uint32_t FlashAttentionScoreGradTilingNormalRegbase::GetDeterSparseTilingKey()
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

uint8_t FlashAttentionScoreGradTilingNormalRegbase::GetSparseType()
{
    // DENSE: 1）非sparse；2）ALL_MASK；3）NO_MASK & preToken>=Sq & nextToken>=Skv
    bool denseCondition = !fBaseParams.isSparse ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token >= fBaseParams.s2);

    bool casualCondition = false;
    bool bandCondition = false;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        casualCondition = (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0);
        // 仅支持N1为偶数，分核时按照N1维度拼接性能最优，如果N1为奇数存在负载不均问题。
        casualCondition = casualCondition && ((fBaseParams.n1 % MULT_BASE == 0));
        if (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            casualCondition = casualCondition && tndBaseInfo.isS1GreaterThanS2;
        } else {
            casualCondition = casualCondition && tndBaseInfo.isS1LessThanS2;
            if (!fBaseParams.isS1S2Same) {
                casualCondition = casualCondition && (fBaseParams.s1 % fBaseParams.s1CvInner == 0);
            }
        }

        bandCondition = fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND);
        bandCondition = bandCondition && fBaseParams.s1Token >= 0 && fBaseParams.s2Token >= 0 && fBaseParams.isS1S2Same;
    } else {
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
    }
    OP_LOGI(
        "GetSparseType",
        "denseCondition = %d, casualCondition = %d, bandCondition = %d, isS1GreaterThanS2 = %d, isS1LessThanS2 = %d",
        denseCondition, casualCondition, bandCondition, tndBaseInfo.isS1GreaterThanS2, tndBaseInfo.isS1LessThanS2);

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

void FlashAttentionScoreGradTilingNormalRegbase::CalcleDeterParam()
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
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        CalcTNDSwizzleParam();
        CalcleTNDDeterParam();
    }
    if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
        fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL)) {
        CalcleCausalDeterParam(fBaseParams);
    } else if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
               fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
        CalcleBandDeterParam(fBaseParams);
    } else if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
               fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_DENSE) &&
               fBaseParams.g > 1) {
        const GQADenseScheduleResult schedule =
            SelectGQADenseSchedule(static_cast<int64_t>(fBaseParams.aicNum), fBaseParams.s1Outer, fBaseParams.s2Outer,
                                   fBaseParams.b * fBaseParams.n2, fBaseParams.g);
        fBaseParams.deterMaxRound = schedule.selectedRound * fBaseParams.s1Outer;
        OP_LOGI(context_,
                "Select deterministic GQA DENSE round, baseR=[%ld], selectedR=[%ld], period=[%ld->%ld], "
                "maxRound=[%ld]",
                schedule.baseRound, schedule.selectedRound, schedule.basePeriod, schedule.selectedPeriod,
                fBaseParams.deterMaxRound);
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

void FlashAttentionScoreGradTilingNormalRegbase::GetIsDeterArr()
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

bool FlashAttentionScoreGradTilingNormalRegbase::CheckExceedL2Cache()
{
    std::array<int64_t, CORE_LIST_NUM> dqOffset;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffset;
    std::array<int64_t, CORE_LIST_NUM> loopIdx;
    std::set<int> dqOffsetSet;
    std::set<int> dkDvOffsetSet;
    uint64_t usedl2CacheSize = 0;
    int64_t calcNum = 0;
    int32_t inputSize = FP16_BYTES;
    std::fill(std::begin(loopIdx), std::end(loopIdx), static_cast<int64_t>(0));

    if (fBaseParams.queryType == ge::DT_FLOAT) {
        inputSize = FP32_BYTES;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        inputSize = FP16_BYTES;
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
               fBaseParams.queryType == ge::DT_HIFLOAT8) {
        inputSize = 1;
    }

    bool isExceed = false;
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

            if (dqOffsetSet.find(dqOffset[cBlockIdx]) == dqOffsetSet.end()) {
                dqOffsetSet.insert(dqOffset[cBlockIdx]);
                // qSize + dxSize + dqSize(btyes) + ySize，3 means q, dx and y
                usedl2CacheSize += (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d * inputSize * NUM_THREE +
                                    fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d * FP32_BYTES);
            }
            if (dkDvOffsetSet.find(dkDvOffset[cBlockIdx]) == dkDvOffsetSet.end()) {
                dkDvOffsetSet.insert(dkDvOffset[cBlockIdx]);
                // kSize + vSize + dkSize + dvSize(btyes)，2 means k/dk and v/dv
                usedl2CacheSize += (fBaseParams.s2Inner * fBaseParams.d * inputSize * NUM_TWO +
                                    fBaseParams.s2Inner * fBaseParams.d * FP32_BYTES * NUM_TWO);
            }
            if (usedl2CacheSize > fBaseParams.l2CacheSize) {
                isExceed = true;
                break;
            }
        }
        if (isExceed) {
            break;
        }
        calcNum++;
    }
    if (!isExceed) {
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
            fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
            if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBN2GS1S2)) {
                usedl2CacheSize += (fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.s2);
            } else if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBS1S2)) {
                usedl2CacheSize += (fBaseParams.b * fBaseParams.s1 * fBaseParams.s2);
            } else if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKS1S2)) {
                usedl2CacheSize += (fBaseParams.s1 * fBaseParams.s2);
            }
        } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
                   fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
                   fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) {
            usedl2CacheSize += COMPRESS_ATTEN_MASK_SIZE;
        }
        isExceed = usedl2CacheSize > fBaseParams.l2CacheSize;
    }

    return isExceed;
}

uint64_t FlashAttentionScoreGradTilingNormalRegbase::DoPreSfmgTiling()
{
    uint32_t valueDAlign = fBaseParams.sfmgdInner;

    uint64_t normalAxisSize = 0;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        normalAxisSize = fBaseParams.t1 * fBaseParams.n2 * fBaseParams.g;
    } else {
        normalAxisSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1;
    }

    int32_t inputSize = FP16_BYTES;
    int32_t outDtypeSize = FP16_BYTES;
    // 计算单loop的计算量及loop次数, hifp8场景按128对齐, quantblock大小为128 * 4, 目前仅支持D <= 256
    uint64_t singleLoopNBurstNum = SFMG_DEFAULT_BURST_NUM;
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        inputSize = FP32_BYTES;
        outDtypeSize = FP32_BYTES;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        inputSize = FP16_BYTES;
        outDtypeSize = FP16_BYTES;
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
               fBaseParams.queryType == ge::DT_HIFLOAT8) {
        inputSize = 1;
        outDtypeSize = FP16_BYTES;
    }
    uint32_t availUbSize = fBaseParams.ubSize - UB_RESERVE_SPACE;
    // valueDAlign * inputSize * sizeof(dtype) * 2 * 2 --  dy, y size is valueDAlign * inputSize
    // first 2 is dy + y total size, second 2 is double buffer, then get max split s1
    uint32_t sfmgDyBufferLen = availUbSize /
                               (valueDAlign * (inputSize * NUM_TWO + outDtypeSize * NUM_TWO) +
                                SFMG_DOUBLE_BUFFER_NUM * SFMG_FP32_ACCUMULATOR_SIZE * FP32_BYTES) *
                               valueDAlign * inputSize;
    uint32_t sfmgYBufferLen = availUbSize /
                              (valueDAlign * (inputSize * NUM_TWO + outDtypeSize * NUM_TWO) +
                               SFMG_DOUBLE_BUFFER_NUM * SFMG_FP32_ACCUMULATOR_SIZE * FP32_BYTES) *
                              valueDAlign * outDtypeSize;
    uint32_t sfmgOutputBufferLen = availUbSize /
                                   (valueDAlign * (inputSize * NUM_TWO + outDtypeSize * NUM_TWO) +
                                    SFMG_DOUBLE_BUFFER_NUM * SFMG_FP32_ACCUMULATOR_SIZE * FP32_BYTES) *
                                   SFMG_FP32_ACCUMULATOR_SIZE * FP32_BYTES;

    // 计算单核的计算量
    uint32_t sfmgUsedCoreNum = fBaseParams.blockOuter * AICV_RATIO_DEFAULT;
    uint64_t normalCoreSize = CeilDivideBy(normalAxisSize, static_cast<uint64_t>(sfmgUsedCoreNum));
    sfmgUsedCoreNum = CeilDivideBy(normalAxisSize, normalCoreSize);
    uint64_t tailCoreSize = normalAxisSize - (sfmgUsedCoreNum - 1) * normalCoreSize;
    // 非fp8场景按照实际head dim的大小计算
    if (fBaseParams.queryType == ge::DT_FLOAT16 || fBaseParams.queryType == ge::DT_BF16) {
        singleLoopNBurstNum = sfmgDyBufferLen / inputSize / valueDAlign;
    }
    uint64_t normalCoreLoopTimes = CeilDivideBy(normalCoreSize, singleLoopNBurstNum);
    uint64_t normalCoreLastLoopNBurstNum = normalCoreSize - (normalCoreLoopTimes - 1) * singleLoopNBurstNum;
    uint64_t tailCoreLoopTimes = CeilDivideBy(tailCoreSize, singleLoopNBurstNum);
    uint64_t tailCoreLastLoopNBurstNum = tailCoreSize - (tailCoreLoopTimes - 1) * singleLoopNBurstNum;

    OP_LOGI("DoPreSfmgTiling",
            "DoPreSfmgTiling, sfmgUsedCoreNum = %d, ubsize = %d, valueDAlign = %d,"
            "normalAxisSize = %d, reals1percore = %d, sfmgDyBufferLen is %d, sfmgYBufferLen is %d, sfmgOutputBufferLen "
            "is %d."
            "singleLoopNBurstNum = %d, normalCoreLoopTimes is %d, normalCoreLastLoopNBurstNum is %d."
            "tailCoreLoopTimes = %d, tailCoreLastLoopNBurstNum is %d.",
            sfmgUsedCoreNum, availUbSize, valueDAlign, normalAxisSize, normalCoreSize, sfmgDyBufferLen, sfmgYBufferLen,
            sfmgOutputBufferLen, singleLoopNBurstNum, normalCoreLoopTimes, normalCoreLastLoopNBurstNum,
            tailCoreLoopTimes, tailCoreLastLoopNBurstNum);
    preTilingData_->set_sfmgUsedCoreNum(sfmgUsedCoreNum);
    preTilingData_->set_sfmgDyBufferLen(sfmgDyBufferLen);
    preTilingData_->set_sfmgYBufferLen(sfmgYBufferLen);
    preTilingData_->set_sfmgOutputBufferLen(sfmgOutputBufferLen);

    preTilingData_->set_singleLoopNBurstNum(singleLoopNBurstNum);
    preTilingData_->set_normalCoreLoopTimes(normalCoreLoopTimes);
    preTilingData_->set_tailCoreLoopTimes(tailCoreLoopTimes);
    preTilingData_->set_normalCoreLastLoopNBurstNum(normalCoreLastLoopNBurstNum);
    preTilingData_->set_tailCoreLastLoopNBurstNum(tailCoreLastLoopNBurstNum);
    preTilingData_->set_normalCoreNBurstNums(normalCoreSize);
    preTilingData_->set_tailCoreNBurstNums(tailCoreSize);

    preTilingData_->set_normalAxisSize(normalAxisSize);
    return sfmgUsedCoreNum;
}

void FlashAttentionScoreGradTilingNormalRegbase::DoPreTiling()
{
    uint64_t inputBufferLen = PRE_BUFFER_SIZE; // x / 8 + 2 * x + 32 = fBaseParams.ubSize
    uint64_t singleUBProcessNum = static_cast<uint64_t>(CAST_BUFFER_LEN) / 2;

    uint64_t maskSize = AlignTo(fBaseParams.dropMaskSize, static_cast<uint64_t>(BOOL_BLOCK_NUMS));
    uint64_t singleCoreNum = AlignTo(CeilDivideBy(maskSize, static_cast<uint64_t>(fBaseParams.blockOuter)),
                                     static_cast<uint64_t>(BOOL_BLOCK_NUMS));
    uint64_t maskUsedCoreNum = 0;
    bool presfmgLimit = !(fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM128) &&
                          fBaseParams.s2 <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM256) &&
                          fBaseParams.b * fBaseParams.n1 * fBaseParams.s1Outer >= MAX_BASIC_BLOCK_SIZE);
    bool isNewDeterForPreSfmg = fBaseParams.deterSparseType >= static_cast<uint32_t>(DeterSparseType::DETER_DENSE) &&
                                fBaseParams.deterSparseType <= static_cast<uint32_t>(DeterSparseType::DETER_BAND);
    bool deterSupportPreSfmg = !fBaseParams.isDeterministic || isNewDeterForPreSfmg;
    fBaseParams.enablePreSfmg =
        deterSupportPreSfmg &&
        ((fBaseParams.queryType == ge::DT_HIFLOAT8) ||
         ((fBaseParams.queryType == ge::DT_BF16 || fBaseParams.queryType == ge::DT_FLOAT16) && presfmgLimit &&
          fBaseParams.d > static_cast<uint32_t>(ConstAxisTemplateNum::NUM64) &&
          fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM768) &&
          (fBaseParams.splitAxis == SplitAxisEnum::BN2GS1S2 || fBaseParams.splitAxis == SplitAxisEnum::BN2S2) &&
          fBaseParams.sinkOptional != NORMAL_TENSOR && fBaseParams.dropoutIsDivisibleBy8 &&
          !fBaseParams.sValueZeroUnderTND));
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

    preTilingData_->set_maskCoreNum(maskUsedCoreNum);
    preTilingData_->set_castBufferLen(CAST_BUFFER_LEN);
    preTilingData_->set_outputBufferLen(OUTPUT_BUFFER_LEN);
    preTilingData_->set_inputBufferLen(inputBufferLen);
    preTilingData_->set_singleUBProcessNum(static_cast<uint32_t>(singleUBProcessNum));
    preTilingData_->set_maskSingleCoreNum(singleCoreNum); // size == num
    preTilingData_->set_maskSingleCoreLoop(singleCoreUBLoop);
    preTilingData_->set_maskLastLoopNum(singleCoreUBLastLoopNum);
    preTilingData_->set_maskTailCoreLoop(tailCoreUBLoop);
    preTilingData_->set_maskTailCoreLastLoopNum(tailCoreUBLastLoopNum);

    uint64_t qSize = fBaseParams.qSize;
    uint64_t kSize = fBaseParams.kSize;
    uint64_t vSize = fBaseParams.vSize;
    if (fBaseParams.isNzOut) {
        qSize = fBaseParams.qSize / fBaseParams.d * AlignTo(fBaseParams.d, static_cast<int64_t>(FP16_C0_SIZE));
        kSize = fBaseParams.kSize / fBaseParams.d * AlignTo(fBaseParams.d, static_cast<int64_t>(FP16_C0_SIZE));
        vSize = fBaseParams.vSize / fBaseParams.d1 * AlignTo(fBaseParams.d1, static_cast<int64_t>(FP16_C0_SIZE));
    }
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

    if (fBaseParams.sinkOptional == NORMAL_TENSOR) {
        fBaseParams.s1SinkOuter = fBaseParams.s1Outer * AICV_RATIO_DEFAULT;
        fBaseParams.s2SinkOuter = fBaseParams.s2Outer;
        fBaseParams.sinkSize = (fBaseParams.b - fBaseParams.tailZeroCount) * fBaseParams.n2 * fBaseParams.g *
                               fBaseParams.s1SinkOuter * fBaseParams.s2SinkOuter;
        uint64_t sinkWorkSpaceSize = (fBaseParams.sinkSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        uint64_t sinkPreBlockFactor = (sinkWorkSpaceSize + maskUsedCoreNum - 1) / maskUsedCoreNum;
        uint64_t sinkPreBlockTotal = (sinkWorkSpaceSize + sinkPreBlockFactor - 1) / sinkPreBlockFactor;
        uint64_t sinkPreTailNumTmp = sinkWorkSpaceSize % sinkPreBlockFactor;
        uint64_t sinkPreTailNum =
            sinkPreTailNumTmp == static_cast<uint64_t>(0) ? sinkPreBlockFactor : sinkPreTailNumTmp;
        preTilingData_->set_sinkPreBlockFactor(sinkPreBlockFactor);
        preTilingData_->set_sinkPreBlockTotal(sinkPreBlockTotal);
        preTilingData_->set_sinkPreBlockTail(sinkPreTailNum);
        OP_LOGI(
            context_,
            "FAG sinkOptional, fBaseParams.s1SinkOuter is %ld, fBaseParams.s2SinkOuter = %ld, fBaseParams.sinkSize = "
            "%ld, maskUsedCoreNum = %ld, sinkPreBlockFactor = %ld, sinkPreBlockTotal = %ld, sinkPreTailNum = %ld.",
            fBaseParams.s1SinkOuter, fBaseParams.s2SinkOuter, fBaseParams.sinkSize, maskUsedCoreNum, sinkPreBlockFactor,
            sinkPreBlockTotal, sinkPreTailNum);
    }

    uint64_t maskPreBlockTotal = fBaseParams.dropMaskSize;
    preTilingData_->set_qPreBlockFactor(qPreBlockFactor);
    preTilingData_->set_qPreBlockTotal(qPreBlockTotal);
    preTilingData_->set_qPreBlockTail(qPreTailNum);
    preTilingData_->set_kPreBlockFactor(kPreBlockFactor);
    preTilingData_->set_kPreBlockTotal(kPreBlockTotal);
    preTilingData_->set_kPreBlockTail(kPreTailNum);
    preTilingData_->set_vPreBlockFactor(vPreBlockFactor);
    preTilingData_->set_vPreBlockTotal(vPreBlockTotal);
    preTilingData_->set_vPreBlockTail(vPreTailNum);
    preTilingData_->set_dropoutIsDivisibleBy8(fBaseParams.dropoutIsDivisibleBy8);
    preTilingData_->set_sValueZeroUnderTND(fBaseParams.sValueZeroUnderTND);
    preTilingData_->set_hasInvalidCol(fBaseParams.isInvalidCol);
}

void FlashAttentionScoreGradTilingNormalRegbase::DoPostTiling()
{
    uint64_t postUbBaseSize = fBaseParams.hasRope ? ROPE_POST_BASE * FP16_BYTES : REGBASE_POST_BASE * FP16_BYTES;
    uint64_t qPostBaseNum = fBaseParams.hasRope ? ROPE_POST_BASE : REGBASE_POST_BASE;
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
    if (fBaseParams.sinkOptional == NORMAL_TENSOR) {
        uint64_t sinkPostBaseNum = postUbBaseSize / FP16_BYTES;
        uint64_t sinkReduceAxis =
            (fBaseParams.b - fBaseParams.tailZeroCount) * fBaseParams.s1SinkOuter * fBaseParams.s2SinkOuter;
        uint64_t sinkPostTailNumTmp = sinkReduceAxis % sinkPostBaseNum;
        uint64_t sinkPostTailNum =
            sinkPostTailNumTmp == static_cast<uint64_t>(0) ? sinkPostBaseNum : sinkPostTailNumTmp;
        uint64_t sinkPostBlockTotal = fBaseParams.n1;
        uint64_t sinkPostBlockFactor = (fBaseParams.n1 + fBaseParams.blockOuter * AICV_RATIO_DEFAULT - 1) /
                                       (fBaseParams.blockOuter * AICV_RATIO_DEFAULT);
        postTilingData_->set_sinkReduceAxis(sinkReduceAxis);
        postTilingData_->set_sinkPostBlockTotal(sinkPostBlockTotal);
        postTilingData_->set_sinkPostBlockFactor(sinkPostBlockFactor);
        postTilingData_->set_sinkPostTailNum(sinkPostTailNum);
    }

    postTilingData_->set_qPostBlockFactor(qPostBlockFactor);
    postTilingData_->set_qPostBlockTotal(qPostBlockTotal);
    postTilingData_->set_qPostBaseNum(qPostBaseNum);
    postTilingData_->set_qPostTailNum(qPostTailNum);

    postTilingData_->set_kPostBlockFactor(kPostBlockFactor);
    postTilingData_->set_kPostBlockTotal(kPostBlockTotal);
    postTilingData_->set_kPostBaseNum(kPostBaseNum);
    postTilingData_->set_kPostTailNum(kPostTailNum);

    postTilingData_->set_vPostBlockFactor(vPostBlockFactor);
    postTilingData_->set_vPostBlockTotal(vPostBlockTotal);
    postTilingData_->set_vPostBaseNum(vPostBaseNum);
    postTilingData_->set_vPostTailNum(vPostTailNum);
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::GetWorkspaceSize()
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
    // dqkv fixp nz出需要n轴16对齐
    if (fBaseParams.isNzOut) {
        qSize = ((fBaseParams.b * fBaseParams.n1 - 1) * fBaseParams.s1 + AlignTo(fBaseParams.s1, ALIGN128)) *
                AlignTo(fBaseParams.d, static_cast<int64_t>(FP16_C0_SIZE));
        kSize = ((fBaseParams.b * fBaseParams.n2 - 1) * fBaseParams.s2 + AlignTo(fBaseParams.s2, ALIGN128)) *
                AlignTo(fBaseParams.d, static_cast<int64_t>(FP16_C0_SIZE));
        vSize = ((fBaseParams.b * fBaseParams.n2 - 1) * fBaseParams.s2 + AlignTo(fBaseParams.s2, ALIGN128)) *
                AlignTo(fBaseParams.d1, static_cast<int64_t>(FP16_C0_SIZE));
    }
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        qSize = (AlignTo(fBaseParams.t1 * fBaseParams.n1, ALIGN128)) * fBaseParams.d;
        kSize = (AlignTo(fBaseParams.t2 * fBaseParams.n2, ALIGN128)) * fBaseParams.d;
        vSize = (AlignTo(fBaseParams.t2 * fBaseParams.n2, ALIGN128)) * fBaseParams.d1;
        if (fBaseParams.isNzOut) {
            qSize = (AlignTo(fBaseParams.t1 * fBaseParams.n1, ALIGN128)) *
                    AlignTo(fBaseParams.d, static_cast<int64_t>(FP16_C0_SIZE));
            kSize = (AlignTo(fBaseParams.t2 * fBaseParams.n2, ALIGN128)) *
                    AlignTo(fBaseParams.d, static_cast<int64_t>(FP16_C0_SIZE));
            vSize = (AlignTo(fBaseParams.t2 * fBaseParams.n2, ALIGN128)) *
                    AlignTo(fBaseParams.d1, static_cast<int64_t>(FP16_C0_SIZE));
        }
    }
    if (fBaseParams.splitAxis == SplitAxisEnum::BN2S2) {
        postTilingData_->set_dqWorkSpaceOffset(workspaceSize);
        // matmal3 q
        workspaceSize += (static_cast<size_t>(qSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        postTilingData_->set_dkWorkSpaceOffset(workspaceSize);
        // matmal3 k
        workspaceSize += (fBaseParams.s2Inner * fBaseParams.sfmgdInner * CORE_LIST_NUM * FP32_BYTES + GM_ALIGN) /
                         GM_ALIGN * GM_ALIGN;
        postTilingData_->set_dvWorkSpaceOffset(workspaceSize);
        // matmal3 v
        workspaceSize += (fBaseParams.s2Inner * fBaseParams.sfmgdInner * CORE_LIST_NUM * FP32_BYTES + GM_ALIGN) /
                         GM_ALIGN * GM_ALIGN;
    } else if (fBaseParams.isBn2) {
        if (fBaseParams.isBn2MultiBlk) {
            postTilingData_->set_dqWorkSpaceOffset(workspaceSize);
            // matmal3 dq
            workspaceSize += CORE_LIST_NUM * (AlignTo(fBaseParams.s1, ALIGN128) * fBaseParams.sfmgdInner * FP32_BYTES);
            postTilingData_->set_dkWorkSpaceOffset(workspaceSize);
            // matmal4 dk
            workspaceSize += CORE_LIST_NUM * fBaseParams.s2Inner * fBaseParams.sfmgdInner * FP32_BYTES;
            // matmal5 dv
            postTilingData_->set_dvWorkSpaceOffset(workspaceSize);
            workspaceSize += CORE_LIST_NUM * fBaseParams.s2Inner * fBaseParams.sfmgdInner * FP32_BYTES;
        } else {
            postTilingData_->set_dqWorkSpaceOffset(workspaceSize);
            workspaceSize +=
                (fBaseParams.s2Inner * fBaseParams.sfmgdInner * NUM_TWO * CORE_LIST_NUM * FP32_BYTES + GM_ALIGN) /
                GM_ALIGN * GM_ALIGN;
        }
    } else {
        if (fBaseParams.queryType != ge::DT_FLOAT) {
            postTilingData_->set_dqWorkSpaceOffset(workspaceSize);
            // matmal3 q
            workspaceSize = (workspaceSize + static_cast<size_t>(qSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
            postTilingData_->set_dkWorkSpaceOffset(workspaceSize);
            // matmal3 k
            workspaceSize = (workspaceSize + static_cast<size_t>(kSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
            postTilingData_->set_dvWorkSpaceOffset(workspaceSize);
            // matmal3 v
            workspaceSize = (workspaceSize + static_cast<size_t>(vSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        }
        // fp8 vScaleDs
        if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
            fBaseParams.queryType == ge::DT_HIFLOAT8) {
            workspaceSize =
                (workspaceSize + fBaseParams.coreNum * ALIGN128 * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        }
    }
    // mask bool workspace size
    if (fBaseParams.dropoutIsDivisibleBy8 == 0 && fBaseParams.dropMaskOuter) {
        postTilingData_->set_dropMaskGmOffset(workspaceSize);
        workspaceSize =
            (workspaceSize + static_cast<size_t>(fBaseParams.dropMaskSize) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }

    if (fBaseParams.enablePreSfmg) {
        // softmax grad workspace size
        postTilingData_->set_sfmgWorkSpaceOffset(workspaceSize);
        uint64_t sfmgSize = ((fBaseParams.b * fBaseParams.n2 * fBaseParams.g - 1) * fBaseParams.s1 +
                             AlignTo(fBaseParams.s1, ALIGN128)) *
                            BIT_NUMS;
        workspaceSize = (workspaceSize + static_cast<size_t>(sfmgSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }

    if (fBaseParams.sinkOptional == NORMAL_TENSOR) {
        postTilingData_->set_dsinkWorkSpaceOffset(workspaceSize);
        OP_LOGI(context_, "FAG sinkOptional, sink baseoffset = %ld, sink workspaceSize = %ld.", workspaceSize,
                static_cast<size_t>(fBaseParams.sinkSize) * FP32_BYTES);
        // dsink sum data size
        workspaceSize =
            (workspaceSize + static_cast<size_t>(fBaseParams.sinkSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }

    GetWorkspaceSize4Deter(workspaceSize);

    workspaceSize += WORKSPACE_BUFFER;
    workspaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingNormalRegbase::GetWorkspaceSize4Deter(size_t &workspaceSize)
{
    if (fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_OLD)) {
        baseDeterParam_->set_deterGmOffset(workspaceSize);
        workspaceSize += (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT + NUM_TWO * fBaseParams.s2Inner) *
                         fBaseParams.sfmgdInner * fBaseParams.aicNum * FP32_BYTES * NUM_TWO;
        baseDeterParam_->set_deterWorkSpaceOffset(workspaceSize);
        // NUM_THREE: querGmOffset, keyGmOffset and valueGmOffset
        workspaceSize += fBaseParams.maxValidBBLen * fBaseParams.aicNum * INT64_BLOCK_NUM * NUM_THREE * INT64_BYTES;
    }

    if (fBaseParams.splitAxis == SplitAxisEnum::BN2S2 &&
        (fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) ||
         fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_DENSE))) {
        baseDeterParam_->set_deterGmOffset(workspaceSize);
        workspaceSize += (fBaseParams.s2Inner * fBaseParams.sfmgdInner * CORE_LIST_NUM * FP32_BYTES + GM_ALIGN) /
                         GM_ALIGN * GM_ALIGN * NUM_TWO;
    }
}

uint64_t FlashAttentionScoreGradTilingNormalRegbase::GetTilingKey() const
{
    auto attenMaskCfg = fBaseParams.attenMaskOptional == EMPTY_TENSOR ? OptionEnum::DISABLE : OptionEnum::ENABLE;
    auto dNoEqual = (fBaseParams.d1 != fBaseParams.d) || fBaseParams.hasRope;
    auto pseValue = fBaseParams.pseOptional == NORMAL_TENSOR ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto dropValue = fBaseParams.keepProb < 1 ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto isRegbasePlatformValue = OptionEnum::ENABLE;
    auto isTnd = (fBaseParams.layoutType == INPUT_FORMAT_TND);
    auto splitAxis = fBaseParams.splitAxis;
    bool isDeterNEqual = fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_OLD) &&
                         fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::NO_DETER) &&
                         fBaseParams.g == 1;
    OP_LOGI(
        context_,
        "splitAxis[%d], inputDtype[%d], isTnd[%d], dropValue[%d], pseValue[%d], attenMaskCfg[%d], s1TemplateType[%d], "
        "s2TemplateType[%d], dTemplateType[%u], isDeterministic[%d], nEqual[%d], isBn2MultiBlk[%d], dNoEqual[%d], "
        "hasRope[%d], outDtype[%d], isNzOut[%d], isTndSwizzle[%d], isRegbasePlatformValue[%d]",
        static_cast<int>(splitAxis), static_cast<int>(fBaseParams.inputDtype), isTnd, static_cast<int>(dropValue),
        static_cast<int>(pseValue), static_cast<int>(attenMaskCfg), static_cast<int>(fBaseParams.s1TemplateType),
        static_cast<int>(fBaseParams.s2TemplateType), static_cast<uint32_t>(fBaseParams.dTemplateType),
        static_cast<int>(fBaseParams.deterSparseType), static_cast<int>(isDeterNEqual),
        static_cast<int>(fBaseParams.isBn2MultiBlk), dNoEqual, static_cast<int>(fBaseParams.hasRope),
        static_cast<int>(fBaseParams.outDtype), static_cast<int>(fBaseParams.isNzOut),
        static_cast<uint8_t>(tndBaseInfo.isTndSwizzle), static_cast<int>(isRegbasePlatformValue));

    uint64_t tilingKey = GET_TPL_TILING_KEY(
        0, static_cast<uint8_t>(splitAxis), static_cast<uint8_t>(fBaseParams.inputDtype), static_cast<uint8_t>(isTnd),
        static_cast<uint8_t>(dropValue), static_cast<uint8_t>(pseValue), static_cast<uint8_t>(attenMaskCfg),
        static_cast<uint16_t>(fBaseParams.s1TemplateType), static_cast<uint16_t>(fBaseParams.s2TemplateType),
        static_cast<uint16_t>(fBaseParams.dTemplateType), static_cast<uint8_t>(fBaseParams.deterSparseType),
        static_cast<uint8_t>(isDeterNEqual), static_cast<uint8_t>(fBaseParams.isBn2MultiBlk),
        static_cast<uint8_t>(dNoEqual), static_cast<uint8_t>(fBaseParams.hasRope),
        static_cast<uint8_t>(fBaseParams.outDtype), static_cast<uint8_t>(fBaseParams.isNzOut),
        static_cast<uint8_t>(tndBaseInfo.isTndSwizzle), static_cast<uint8_t>(isRegbasePlatformValue));

    OP_LOGI(context_, "FAGTiling S1s2Bn2gs1s2 DoTiling success, tiling is %lu.", tilingKey);
    return tilingKey;
}

std::tuple<uint32_t, uint32_t, uint32_t> FlashAttentionScoreGradTilingNormalRegbase::FuzzyForBestSplit()
{
    auto s1s2TemplateSize = GetS1S2TemplateType(fBaseParams);
    uint32_t s1Inner = s1s2TemplateSize.first / 2;
    uint32_t s2Inner = s1s2TemplateSize.second;
    uint32_t dInner = GetDTemplateType(fBaseParams);
    return std::tie(s1Inner, s2Inner, dInner);
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::PostTiling()
{
    SaveToTilingData();
    auto numBlocks = 0;
    if (fBaseParams.isDeterministic ||
        (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
         fBaseParams.queryType == ge::DT_HIFLOAT8)) {
        numBlocks = fBaseParams.aicNum;
    } else {
        numBlocks = CalcTschBlockDim(s1s2BNGS1S2SplitCoreParams_->get_blockOuter() * AICV_RATIO_DEFAULT,
                                     fBaseParams.aicNum, fBaseParams.coreNum);
    }
    OP_CHECK_IF(numBlocks == 0,
                OP_LOGE("FlashAttentionScoreGradTilingNormalRegbase", "numBlocks is 0, aicNum is %lu, aivNum is %lu.",
                        fBaseParams.aicNum, fBaseParams.coreNum),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(numBlocks);

    // 使用SyncAll，需要设置为batch mode模式，所有核同时启动，否则在多流方式下执行可能会卡死
    if (fBaseParams.splitAxis != SplitAxisEnum::BN2 || !fBaseParams.isBn2MultiBlk ||
        fBaseParams.layoutType != INPUT_FORMAT_TND) {
        context_->SetScheduleMode(1);
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingNormalRegbase::GetParseS1S2OuterInfo(int64_t (*parseInfo)[ARRAY_LENGTH])
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
        if (parseInfo[i][BEGIN_IDX] >= parseInfo[i][END_IDX]) {
            fBaseParams.isInvalidCol = true;
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
    for (size_t j = 0; j < invalidS1Array.size(); j++) {
        if (!invalidS1Array[j]) {
            fBaseParams.isInvalidRow = true;
            break;
        }
    }
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::GetSparseBlockInfo()
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

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::GetSparsePrefixBlockInfo()
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

void FlashAttentionScoreGradTilingNormalRegbase::FillBlockInfoLoadBalance(
    std::vector<std::vector<int64_t>> &totalBlockInfo, std::vector<std::vector<float>> &acturalBlockInfo)
{
    acturalBlockInfo[fBaseParams.b][0] = 0;
    acturalBlockInfo[fBaseParams.b + 1][0] = 0;
    OP_LOGD("FillBlockInfoLoadBalance", "SparseMode %u, find band index %u", fBaseParams.sparseMode,
            fBaseParams.bandIdx);

    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];

        auto actualS1Outer = (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
        auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;
        // 针对S为0的场景，pre中增加initGm为0的操作
        if (totalBlockInfo[i][0] == 0) {
            fBaseParams.sValueZeroUnderTND = true;
        }

        // 对unpad场景的token值做二次校正
        // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
        int64_t actualCalcS1Token, actualCalcS2Token;
        CalcleActualToken(fBaseParams, i, actualCalcS1Token, actualCalcS2Token);

        OP_LOGD("FillBlockInfoLoadBalance",
                " b idx = %ld: actualS1Len = %ld, actualS2Len = %ld, actualCalcS1Token = %ld, actualCalcS2Token = %ld",
                i, actualS1Len, actualS2Len, actualCalcS1Token, actualCalcS2Token);

        // unpad 场景下s2Outer是按照最大的s2计算得到的
        for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
            if (fBaseParams.cvS2Inner * j >= actualS2Len) {
                acturalBlockInfo[i][j] = 0;
            } else {
                int64_t leftIntersectionPoint = std::max(int64_t(fBaseParams.cvS2Inner * j) - actualCalcS2Token, 0L);
                int64_t cvBlockTail = fBaseParams.cvS2Inner * (j + 1) > actualS2Len ?
                                          actualS2Len - fBaseParams.cvS2Inner * j :
                                          fBaseParams.cvS2Inner;

                float acturalS1Begin = static_cast<float>(
                    leftIntersectionPoint > int64_t(actualS1Len) ? actualS1Len : leftIntersectionPoint);
                float acturalS1End = static_cast<float>(std::min(
                    int64_t(actualS1Len), std::max(fBaseParams.cvS2Inner * j + cvBlockTail + actualCalcS1Token, 0L)));
                if (fBaseParams.splitAxis == SplitAxisEnum::BN2S2 && (acturalS1Begin >= acturalS1End)) {
                    fBaseParams.isInvalidCol = true;
                }
                float acturalS1Num = acturalS1Begin > acturalS1End ? 0 : acturalS1End - acturalS1Begin;
                float acturalS2Num = static_cast<float>(cvBlockTail);
                acturalBlockInfo[i][j] = acturalS1Num / static_cast<float>(fBaseParams.s1CvInner) +
                                         acturalS2Num / static_cast<float>(fBaseParams.cvS2Inner);

                acturalBlockInfo[fBaseParams.b][0] += acturalBlockInfo[i][j] * fBaseParams.n2 * fBaseParams.g;
                acturalBlockInfo[fBaseParams.b + 1][0] =
                    acturalBlockInfo[fBaseParams.b + 1][0] < acturalBlockInfo[i][j] ?
                        acturalBlockInfo[i][j] :
                        acturalBlockInfo[fBaseParams.b + 1][0];
            }
        }

        if (i == 0) {
            totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
        } else {
            totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::InitTilingData()
{
    bool isTnd = (fBaseParams.layoutType == INPUT_FORMAT_TND);
    if (IsNewDeter(fBaseParams) && tndBaseInfo.isTndSwizzle) {
        FagTilingWithTemplateTTTT *tilingData = this->context_->GetTilingData<FagTilingWithTemplateTTTT>();
        if (tilingData == nullptr) {
            OP_LOGE("InitTilingData", "InitTilingData failed.");
            return ge::GRAPH_FAILED;
        }
        s1s2BNGS1S2BaseParams_ = &tilingData->s1s2BNGS1S2BaseParams;
        s1s2BNGS1S2SplitCoreParams_ = &tilingData->s1s2BNGS1S2SplitCoreParams;
        s1s2BNGS1S2BlockNumList_ = &tilingData->s1s2BNGS1S2BlockNumList;
        preTilingData_ = &tilingData->preTilingData;
        postTilingData_ = &tilingData->postTilingData;
        deterParam = &tilingData->deterParam;
        tndSwizzleParam_ = &tilingData->tndSwizzleParam;
        baseDeterParam_ = &tilingData->baseDeterParam;
    } else if (IsNewDeter(fBaseParams)) {
        FagTilingWithTemplateTTTF *tilingData = this->context_->GetTilingData<FagTilingWithTemplateTTTF>();
        if (tilingData == nullptr) {
            OP_LOGE("InitTilingData", "InitTilingData failed.");
            return ge::GRAPH_FAILED;
        }
        s1s2BNGS1S2BaseParams_ = &tilingData->s1s2BNGS1S2BaseParams;
        s1s2BNGS1S2SplitCoreParams_ = &tilingData->s1s2BNGS1S2SplitCoreParams;
        s1s2BNGS1S2BlockNumList_ = &tilingData->s1s2BNGS1S2BlockNumList;
        preTilingData_ = &tilingData->preTilingData;
        postTilingData_ = &tilingData->postTilingData;
        deterParam = &tilingData->deterParam;
        baseDeterParam_ = &tilingData->baseDeterParam;
    } else if (tndBaseInfo.isTndSwizzle) {
        FagTilingWithTemplateFFTT *tilingData = this->context_->GetTilingData<FagTilingWithTemplateFFTT>();
        if (tilingData == nullptr) {
            OP_LOGE("InitTilingData", "InitTilingData failed.");
            return ge::GRAPH_FAILED;
        }
        s1s2BNGS1S2BaseParams_ = &tilingData->s1s2BNGS1S2BaseParams;
        s1s2BNGS1S2SplitCoreParams_ = &tilingData->s1s2BNGS1S2SplitCoreParams;
        s1s2BNGS1S2BlockNumList_ = &tilingData->s1s2BNGS1S2BlockNumList;
        preTilingData_ = &tilingData->preTilingData;
        postTilingData_ = &tilingData->postTilingData;
        tndSwizzleParam_ = &tilingData->tndSwizzleParam;
    } else if (isTnd) {
        if (fBaseParams.isDeterministic) {
            FagTilingWithTemplateTFTF *tilingData = this->context_->GetTilingData<FagTilingWithTemplateTFTF>();
            if (tilingData == nullptr) {
                OP_LOGE("InitTilingData", "InitTilingData failed.");
                return ge::GRAPH_FAILED;
            }
            TND_TILING_DATA_COMMON_ASSIGN(tilingData);
            baseDeterParam_ = &tilingData->baseDeterParam;
        } else {
            FagTilingWithTemplateFFTF *tilingData = this->context_->GetTilingData<FagTilingWithTemplateFFTF>();
            if (tilingData == nullptr) {
                OP_LOGE("InitTilingData", "InitTilingData failed.");
                return ge::GRAPH_FAILED;
            }
            TND_TILING_DATA_COMMON_ASSIGN(tilingData);
        }
    } else {
        if (fBaseParams.isDeterministic) {
            FagTilingWithTemplateTFFF *tilingData = this->context_->GetTilingData<FagTilingWithTemplateTFFF>();
            if (tilingData == nullptr) {
                OP_LOGE("InitTilingData", "InitTilingData failed.");
                return ge::GRAPH_FAILED;
            }
            BASE_TILING_DATA_COMMON_ASSIGN(tilingData);
            baseDeterParam_ = &tilingData->baseDeterParam;
        } else {
            FagTilingWithTemplateFFFF *tilingData = this->context_->GetTilingData<FagTilingWithTemplateFFFF>();
            if (tilingData == nullptr) {
                OP_LOGE("InitTilingData", "InitTilingData failed.");
                return ge::GRAPH_FAILED;
            }
            BASE_TILING_DATA_COMMON_ASSIGN(tilingData);
        }
    }
    if (s1s2BNGS1S2BaseParams_ == nullptr || s1s2BNGS1S2SplitCoreParams_ == nullptr ||
        s1s2BNGS1S2BlockNumList_ == nullptr || preTilingData_ == nullptr || postTilingData_ == nullptr) {
        OP_LOGE("InitTilingData", "InitTilingData failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::SaveToTilingData()
{
    s1s2BNGS1S2BaseParams_->set_coreNum(fBaseParams.coreNum);
    // set tilingdata baseinfo
    s1s2BNGS1S2BaseParams_->set_b(fBaseParams.b - fBaseParams.tailZeroCount);
    s1s2BNGS1S2BaseParams_->set_n2(fBaseParams.n2);
    s1s2BNGS1S2BaseParams_->set_g(fBaseParams.g);
    s1s2BNGS1S2BaseParams_->set_s1(fBaseParams.s1);
    s1s2BNGS1S2BaseParams_->set_d(fBaseParams.d);
    s1s2BNGS1S2BaseParams_->set_d1(fBaseParams.d1);
    s1s2BNGS1S2BaseParams_->set_s2(fBaseParams.s2);
    s1s2BNGS1S2BaseParams_->set_pseType(fBaseParams.pseType);
    s1s2BNGS1S2BaseParams_->set_pseShapeType(fBaseParams.pseShapeType);
    s1s2BNGS1S2BaseParams_->set_pseLayoutType(fBaseParams.pseLayoutType);
    s1s2BNGS1S2BaseParams_->set_pseDtype(fBaseParams.pseDtype);
    s1s2BNGS1S2BaseParams_->set_attenMaskShapeType(fBaseParams.attenMaskShapeType);
    s1s2BNGS1S2BaseParams_->set_layout(fBaseParams.layoutType);
    s1s2BNGS1S2BaseParams_->set_tndMaxSumLayout(fBaseParams.tndMaxSumLayout);
    s1s2BNGS1S2BaseParams_->set_scaleValue(fBaseParams.scaleValue);
    s1s2BNGS1S2BaseParams_->set_keepProb(fBaseParams.keepProb);
    s1s2BNGS1S2BaseParams_->set_keepProbUint8(fBaseParams.keepProbUint8);
    // fBaseParams.s1Token int64_t类型   s1s2BNGS1S2BaseParams_->s1Token  int32_t类型 防止溢出
    s1s2BNGS1S2BaseParams_->set_s1Token(fBaseParams.s1Token > INT32_MAX ? INT32_MAX : fBaseParams.s1Token);
    s1s2BNGS1S2BaseParams_->set_s2Token(fBaseParams.s2Token > INT32_MAX ? INT32_MAX : fBaseParams.s2Token);
    s1s2BNGS1S2BaseParams_->set_sparseMode(fBaseParams.sparseMode);
    s1s2BNGS1S2BaseParams_->set_attenMaskS2Size(fBaseParams.attenMaskS2Size);
    s1s2BNGS1S2BaseParams_->set_attenMaskCompressMode(fBaseParams.attenMaskCompressMode);
    s1s2BNGS1S2BaseParams_->set_seed(fBaseParams.seed);
    s1s2BNGS1S2BaseParams_->set_offset(fBaseParams.offset);
    s1s2BNGS1S2BaseParams_->set_qStartIdx(fBaseParams.qStartIdx);
    s1s2BNGS1S2BaseParams_->set_kvStartIdx(fBaseParams.kvStartIdx);
    s1s2BNGS1S2BaseParams_->set_dropMaskOuter(static_cast<uint8_t>(fBaseParams.dropMaskOuter));
    s1s2BNGS1S2BaseParams_->set_sinkOptional(fBaseParams.sinkOptional);
    s1s2BNGS1S2BaseParams_->set_s1SinkOuter(fBaseParams.s1SinkOuter);
    s1s2BNGS1S2BaseParams_->set_s2SinkOuter(fBaseParams.s2SinkOuter);

    s1s2BNGS1S2BaseParams_->set_isSplitByBlockIdx(static_cast<uint8_t>(fBaseParams.isSplitByBlockIdx));
    s1s2BNGS1S2BaseParams_->set_deterBandScheduleMode(static_cast<uint8_t>(fBaseParams.deterBandScheduleMode));
    if (fBaseParams.isSplitByBlockIdx) {
        s1s2BNGS1S2BaseParams_->set_totalPerBatchNum(GetTotalPerBatchNum(fBaseParams, fBaseParams.sparseType));
    }
    s1s2BNGS1S2BaseParams_->set_sparseType(fBaseParams.sparseType);
    s1s2BNGS1S2BaseParams_->set_enablePreSfmg(fBaseParams.enablePreSfmg);
    // s1/s2 split
    s1s2BNGS1S2SplitCoreParams_->set_s1Outer(fBaseParams.s1Outer);
    s1s2BNGS1S2SplitCoreParams_->set_s1Inner(fBaseParams.s1Inner);
    s1s2BNGS1S2SplitCoreParams_->set_s1CvInner(fBaseParams.s1CvInner);
    s1s2BNGS1S2SplitCoreParams_->set_s1Tail(fBaseParams.s1Tail);
    s1s2BNGS1S2SplitCoreParams_->set_s1CvTail(fBaseParams.s1CvTail);
    s1s2BNGS1S2SplitCoreParams_->set_s2Outer(fBaseParams.s2Outer);
    s1s2BNGS1S2SplitCoreParams_->set_s2Inner(fBaseParams.s2Inner);
    s1s2BNGS1S2SplitCoreParams_->set_s2Tail(fBaseParams.s2Tail);
    s1s2BNGS1S2SplitCoreParams_->set_bandIdx(fBaseParams.bandIdx);
    s1s2BNGS1S2BlockNumList_->set_blockStarts(fBaseParams.blockStarts);
    s1s2BNGS1S2BlockNumList_->set_blockEnds(fBaseParams.blockEnds);
    s1s2BNGS1S2SplitCoreParams_->set_blockOuter(fBaseParams.blockOuter);
    s1s2BNGS1S2SplitCoreParams_->set_maxValidBBLen(fBaseParams.maxValidBBLen);
    if (fBaseParams.isDeterministic) {
        baseDeterParam_->set_noNeedDeter(fBaseParams.noNeedDeter);
        baseDeterParam_->set_deterMaxRound(fBaseParams.deterMaxRound);
        if ((fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) ||
             fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_DENSE)) &&
            fBaseParams.layoutType == INPUT_FORMAT_TND) {
            baseDeterParam_->set_dqIsNeedDeter(fBaseParams.startNeedSyncRound);
            baseDeterParam_->set_dkDvIsNeedDeter(fBaseParams.endNeedSyncRound);
        } else {
            baseDeterParam_->set_dqIsNeedDeter(fBaseParams.dqIsNeedDeter);
            baseDeterParam_->set_dkDvIsNeedDeter(fBaseParams.dkDvIsNeedDeter);
        }
    }
    if (IsNewDeter(fBaseParams) && deterParam != nullptr) {
        deterParam->set_coreDivide(fBaseParams.coreDivide);
        deterParam->set_tndLineDeter(fBaseParams.tndLineDeter);
        deterParam->set_deterPrefixStep(fBaseParams.deterPrefixStep);
        deterParam->set_deterPrefix(fBaseParams.deterPrefix);
        deterParam->set_deterPrefixAlign(fBaseParams.deterPrefixAlign);
        deterParam->set_deterPrefix0(fBaseParams.deterPrefix0);
        deterParam->set_deterPrefix1(fBaseParams.deterPrefix1);
        deterParam->set_deterPrefix2(fBaseParams.deterPrefix2);
    }
    if (tndBaseInfo.isTndSwizzle && tndSwizzleParam_ != nullptr) {
        tndSwizzleParam_->set_tndS2BlockPrefixSum(tndBaseInfo.tndS2BlockPrefixSum);
        tndSwizzleParam_->set_tndSwizzleS1S2PrefixSum(tndBaseInfo.tndSwizzleS1S2PrefixSum);
        tndSwizzleParam_->set_tndSwizzleS1S2AlignPrefixSum(tndBaseInfo.tndSwizzleS1S2AlignPrefixSum);
    } else if (!IsNewDeter(fBaseParams) && fBaseParams.layoutType == INPUT_FORMAT_TND && tndParam_ != nullptr) {
        tndParam_->set_tndStartBIdx(tndBaseInfo.tndStartBIdx);
        tndParam_->set_tndS1S2PrefixSum(tndBaseInfo.tndS1S2PrefixSum);
        tndParam_->set_tndS1S2AlignPrefixSum(tndBaseInfo.tndS1S2AlignPrefixSum);
        tndParam_->set_tndPrefixSum(tndBaseInfo.tndPrefixSum);
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScoreGrad, FlashAttentionScoreGradTilingNormalRegbase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 950);
} // namespace fag
} // namespace optiling
