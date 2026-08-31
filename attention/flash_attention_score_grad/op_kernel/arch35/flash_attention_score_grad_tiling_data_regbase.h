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
 * \file flash_attention_score_grad_tiling_data_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_TILING_DATA_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_TILING_DATA_REGBASE_H_

#include <cstdint>

namespace optiling {
namespace fag {
constexpr uint32_t MAX_CORE_NUM = 36;
constexpr uint32_t TND_SWIZZLE_PREFIX_NUM = 129;

// Schedule selected for the non-TND deterministic BAND implementation. isSplitByBlockIdx keeps
// its original boolean meaning and only controls whether block-index splitting is enabled.
enum class DeterBandScheduleMode : uint8_t {
    DISABLED = 0,
    CAUSAL = 1,
    DENSE = 2,
    BAND = 3
};

class FlashAttentionScoreGradEmptyTensorTilingDataRegbase {
public:
    uint32_t formerDqNum;
    uint32_t formerDkNum;
    uint32_t formerDvNum;
    uint32_t formerDpseNum;
    uint32_t formerDqRopeNum;
    uint32_t formerDkRopeNum;
    uint64_t singleCoreDqNum;
    uint64_t tailCoreDqNum;
    uint64_t singleCoreDkNum;
    uint64_t tailCoreDkNum;
    uint64_t singleCoreDvNum;
    uint64_t tailCoreDvNum;
    uint64_t singleCoreDpseNum;
    uint64_t tailCoreDpseNum;
    uint64_t singleCoreDqRopeNum;
    uint64_t tailCoreDqRopeNum;
    uint64_t singleCoreDkRopeNum;
    uint64_t tailCoreDkRopeNum;
    uint64_t isRope = 0;

    uint32_t get_formerDqNum() const
    {
        return formerDqNum;
    }
    uint32_t get_formerDkNum() const
    {
        return formerDkNum;
    }
    uint32_t get_formerDvNum() const
    {
        return formerDvNum;
    }
    uint32_t get_formerDpseNum() const
    {
        return formerDpseNum;
    }
    uint32_t get_formerDqRopeNum() const
    {
        return formerDqRopeNum;
    }
    uint32_t get_formerDkRopeNum() const
    {
        return formerDkRopeNum;
    }
    uint64_t get_singleCoreDqNum() const
    {
        return singleCoreDqNum;
    }
    uint64_t get_tailCoreDqNum() const
    {
        return tailCoreDqNum;
    }
    uint64_t get_singleCoreDkNum() const
    {
        return singleCoreDkNum;
    }
    uint64_t get_tailCoreDkNum() const
    {
        return tailCoreDkNum;
    }
    uint64_t get_singleCoreDvNum() const
    {
        return singleCoreDvNum;
    }
    uint64_t get_tailCoreDvNum() const
    {
        return tailCoreDvNum;
    }
    uint64_t get_singleCoreDpseNum() const
    {
        return singleCoreDpseNum;
    }
    uint64_t get_tailCoreDpseNum() const
    {
        return tailCoreDpseNum;
    }
    uint64_t get_singleCoreDqRopeNum() const
    {
        return singleCoreDqRopeNum;
    }
    uint64_t get_tailCoreDqRopeNum() const
    {
        return tailCoreDqRopeNum;
    }
    uint64_t get_singleCoreDkRopeNum() const
    {
        return singleCoreDkRopeNum;
    }
    uint64_t get_tailCoreDkRopeNum() const
    {
        return tailCoreDkRopeNum;
    }

    void set_formerDqNum(uint32_t formerDqNumParam)
    {
        this->formerDqNum = formerDqNumParam;
    }
    void set_formerDkNum(uint32_t formerDkNumParam)
    {
        this->formerDkNum = formerDkNumParam;
    }
    void set_formerDvNum(uint32_t formerDvNumParam)
    {
        this->formerDvNum = formerDvNumParam;
    }
    void set_formerDpseNum(uint32_t formerDpseNumParam)
    {
        this->formerDpseNum = formerDpseNumParam;
    }
    void set_formerDqRopeNum(uint32_t formerDqRopeNumParam)
    {
        this->formerDqRopeNum = formerDqRopeNumParam;
    }
    void set_formerDkRopeNum(uint32_t formerDkRopeNumParam)
    {
        this->formerDkRopeNum = formerDkRopeNumParam;
    }
    void set_singleCoreDqNum(uint64_t singleCoreDqNumParam)
    {
        this->singleCoreDqNum = singleCoreDqNumParam;
    }
    void set_tailCoreDqNum(uint64_t tailCoreDqNumParam)
    {
        this->tailCoreDqNum = tailCoreDqNumParam;
    }
    void set_singleCoreDkNum(uint64_t singleCoreDkNumParam)
    {
        this->singleCoreDkNum = singleCoreDkNumParam;
    }
    void set_tailCoreDkNum(uint64_t tailCoreDkNumParam)
    {
        this->tailCoreDkNum = tailCoreDkNumParam;
    }
    void set_singleCoreDvNum(uint64_t singleCoreDvNumParam)
    {
        this->singleCoreDvNum = singleCoreDvNumParam;
    }
    void set_tailCoreDvNum(uint64_t tailCoreDvNumParam)
    {
        this->tailCoreDvNum = tailCoreDvNumParam;
    }
    void set_singleCoreDpseNum(uint64_t singleCoreDpseNumParam)
    {
        this->singleCoreDpseNum = singleCoreDpseNumParam;
    }
    void set_tailCoreDpseNum(uint64_t tailCoreDpseNumParam)
    {
        this->tailCoreDpseNum = tailCoreDpseNumParam;
    }
    void set_singleCoreDqRopeNum(uint64_t singleCoreDqRopeNumParam)
    {
        this->singleCoreDqRopeNum = singleCoreDqRopeNumParam;
    }
    void set_tailCoreDqRopeNum(uint64_t tailCoreDqRopeNumParam)
    {
        this->tailCoreDqRopeNum = tailCoreDqRopeNumParam;
    }
    void set_singleCoreDkRopeNum(uint64_t singleCoreDkRopeNumParam)
    {
        this->singleCoreDkRopeNum = singleCoreDkRopeNumParam;
    }
    void set_tailCoreDkRopeNum(uint64_t tailCoreDkRopeNumParam)
    {
        this->tailCoreDkRopeNum = tailCoreDkRopeNumParam;
    }
};

class FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase {
public:
    int64_t coreNum;
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t d1;
    float scaleValue;
    float keepProb;
    int64_t keepProbUint8;
    int64_t s1Token;
    int64_t s2Token;
    int64_t seed;
    int64_t offset;
    int64_t totalPerBatchNum;
    uint32_t layout;
    uint32_t tndMaxSumLayout;
    uint32_t pseType;
    uint32_t pseShapeType;
    uint32_t pseLayoutType;
    uint32_t pseDtype;
    uint32_t qStartIdx;
    uint32_t kvStartIdx;
    uint32_t attenMaskShapeType;
    uint32_t sparseMode;
    uint32_t attenMaskCompressMode;
    uint32_t attenMaskS2Size;
    uint32_t sinkOptional;
    uint8_t isSplitByBlockIdx;
    uint8_t dropMaskOuter;
    uint8_t sparseType;
    bool enablePreSfmg;
    uint64_t s1SinkOuter;
    uint64_t s2SinkOuter;
    uint8_t deterBandScheduleMode;

    int64_t get_coreNum() const
    {
        return coreNum;
    }
    int64_t get_b() const
    {
        return b;
    }
    int64_t get_n2() const
    {
        return n2;
    }
    int64_t get_g() const
    {
        return g;
    }
    int64_t get_s1() const
    {
        return s1;
    }
    int64_t get_s2() const
    {
        return s2;
    }
    int64_t get_d() const
    {
        return d;
    }
    int64_t get_d1() const
    {
        return d1;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    float get_keepProb() const
    {
        return keepProb;
    }
    int64_t get_keepProbUint8() const
    {
        return keepProbUint8;
    }
    uint8_t get_dropMaskOuter() const
    {
        return dropMaskOuter;
    }
    uint32_t get_layout() const
    {
        return layout;
    }
    uint32_t get_tndMaxSumLayout() const
    {
        return tndMaxSumLayout;
    }
    uint32_t get_pseType() const
    {
        return pseType;
    }
    uint32_t get_qStartIdx() const
    {
        return qStartIdx;
    }
    uint32_t get_kvStartIdx() const
    {
        return kvStartIdx;
    }
    uint32_t get_attenMaskShapeType() const
    {
        return attenMaskShapeType;
    }
    int64_t get_s1Token() const
    {
        return s1Token;
    }
    int64_t get_s2Token() const
    {
        return s2Token;
    }
    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    int64_t get_seed() const
    {
        return seed;
    }
    int64_t get_offset() const
    {
        return offset;
    }
    uint32_t get_attenMaskCompressMode() const
    {
        return attenMaskCompressMode;
    }
    uint32_t get_attenMaskS2Size() const
    {
        return attenMaskS2Size;
    }
    uint8_t get_isSplitByBlockIdx() const
    {
        return isSplitByBlockIdx;
    }
    int64_t get_totalPerBatchNum() const
    {
        return totalPerBatchNum;
    }
    uint8_t get_sparseType() const
    {
        return sparseType;
    }
    bool get_enablePreSfmg() const
    {
        return enablePreSfmg;
    }
    uint32_t get_sinkOptional() const
    {
        return sinkOptional;
    }
    uint64_t get_s1SinkOuter() const
    {
        return s1SinkOuter;
    }
    uint64_t get_s2SinkOuter() const
    {
        return s2SinkOuter;
    }
    uint8_t get_deterBandScheduleMode() const
    {
        return deterBandScheduleMode;
    }

    void set_coreNum(int64_t coreNumParam)
    {
        this->coreNum = coreNumParam;
    }
    void set_b(int64_t bParam)
    {
        this->b = bParam;
    }
    void set_n2(int64_t n2Param)
    {
        this->n2 = n2Param;
    }
    void set_g(int64_t gParam)
    {
        this->g = gParam;
    }
    void set_s1(int64_t s1Param)
    {
        this->s1 = s1Param;
    }
    void set_s2(int64_t s2Param)
    {
        this->s2 = s2Param;
    }
    void set_d(int64_t dParam)
    {
        this->d = dParam;
    }
    void set_d1(int64_t d1Param)
    {
        this->d1 = d1Param;
    }
    void set_scaleValue(float scaleValueParam)
    {
        this->scaleValue = scaleValueParam;
    }
    void set_keepProb(float keepProbParam)
    {
        this->keepProb = keepProbParam;
    }
    void set_keepProbUint8(int64_t keepProbUint8Param)
    {
        this->keepProbUint8 = keepProbUint8Param;
    }
    void set_dropMaskOuter(uint8_t dropMaskOuterParam)
    {
        this->dropMaskOuter = dropMaskOuterParam;
    }
    void set_layout(uint32_t layoutParam)
    {
        this->layout = layoutParam;
    }
    void set_tndMaxSumLayout(uint32_t tndMaxSumLayoutParam)
    {
        this->tndMaxSumLayout = tndMaxSumLayoutParam;
    }
    void set_pseType(uint32_t pseTypeParam)
    {
        this->pseType = pseTypeParam;
    }
    void set_pseShapeType(uint32_t pseShapeTypeParam)
    {
        this->pseShapeType = pseShapeTypeParam;
    }
    void set_pseLayoutType(uint32_t pseLayoutTypeParam)
    {
        this->pseLayoutType = pseLayoutTypeParam;
    }
    void set_pseDtype(uint32_t pseDtypeParam)
    {
        this->pseDtype = pseDtypeParam;
    }
    void set_qStartIdx(uint32_t qStartIdxParam)
    {
        this->qStartIdx = qStartIdxParam;
    }
    void set_kvStartIdx(uint32_t kvStartIdxParam)
    {
        this->kvStartIdx = kvStartIdxParam;
    }
    void set_attenMaskShapeType(uint32_t attenMaskShapeTypeParam)
    {
        this->attenMaskShapeType = attenMaskShapeTypeParam;
    }
    void set_s1Token(int64_t s1TokenParam)
    {
        this->s1Token = s1TokenParam;
    }
    void set_s2Token(int64_t s2TokenParam)
    {
        this->s2Token = s2TokenParam;
    }
    void set_sparseMode(uint32_t sparseModeParam)
    {
        this->sparseMode = sparseModeParam;
    }
    void set_seed(int64_t seedParam)
    {
        this->seed = seedParam;
    }
    void set_offset(int64_t offsetParam)
    {
        this->offset = offsetParam;
    }
    void set_attenMaskCompressMode(uint32_t attenMaskCompressModeParam)
    {
        this->attenMaskCompressMode = attenMaskCompressModeParam;
    }
    void set_attenMaskS2Size(uint32_t attenMaskS2SizeParam)
    {
        this->attenMaskS2Size = attenMaskS2SizeParam;
    }
    void set_isSplitByBlockIdx(uint8_t isSplitByBlockIdxParam)
    {
        this->isSplitByBlockIdx = isSplitByBlockIdxParam;
    }
    void set_totalPerBatchNum(int64_t totalPerBatchNumParam)
    {
        this->totalPerBatchNum = totalPerBatchNumParam;
    }
    void set_sparseType(uint8_t sparseTypeParam)
    {
        this->sparseType = sparseTypeParam;
    }
    void set_enablePreSfmg(uint8_t enablePreSfmgParam)
    {
        this->enablePreSfmg = enablePreSfmgParam;
    }
    void set_sinkOptional(uint32_t sinkOptionalParam)
    {
        this->sinkOptional = sinkOptionalParam;
    }
    void set_s1SinkOuter(uint64_t s1SinkOuterParam)
    {
        this->s1SinkOuter = s1SinkOuterParam;
    }
    void set_s2SinkOuter(uint64_t s2SinkOuterParam)
    {
        this->s2SinkOuter = s2SinkOuterParam;
    }
    void set_deterBandScheduleMode(uint8_t deterBandScheduleModeParam)
    {
        this->deterBandScheduleMode = deterBandScheduleModeParam;
    }
};

class FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsRegbase {
public:
    int64_t s1Outer;
    uint32_t s1Inner;
    uint32_t s1CvInner;
    uint32_t s1Tail;
    uint32_t s1CvTail;
    int64_t s2Outer;
    uint32_t s2Inner;
    uint32_t s2Tail;
    uint32_t blockOuter;
    uint32_t maxValidBBLen;
    int64_t bandIdx;

    int64_t get_s1Outer() const
    {
        return s1Outer;
    }
    uint32_t get_s1Inner() const
    {
        return s1Inner;
    }
    uint32_t get_s1CvInner() const
    {
        return s1CvInner;
    }
    uint32_t get_s1Tail() const
    {
        return s1Tail;
    }
    uint32_t get_s1CvTail() const
    {
        return s1CvTail;
    }
    int64_t get_s2Outer() const
    {
        return s2Outer;
    }
    uint32_t get_s2Inner() const
    {
        return s2Inner;
    }
    uint32_t get_s2Tail() const
    {
        return s2Tail;
    }
    uint32_t get_blockOuter() const
    {
        return blockOuter;
    }
    uint32_t get_maxValidBBLen() const
    {
        return maxValidBBLen;
    }
    int64_t get_bandIdx() const
    {
        return bandIdx;
    }

    void set_s1Outer(int64_t val)
    {
        s1Outer = val;
    }
    void set_s1Inner(uint32_t val)
    {
        s1Inner = val;
    }
    void set_s1CvInner(uint32_t val)
    {
        s1CvInner = val;
    }
    void set_s1Tail(uint32_t val)
    {
        s1Tail = val;
    }
    void set_s1CvTail(uint32_t val)
    {
        s1CvTail = val;
    }
    void set_s2Outer(int64_t val)
    {
        s2Outer = val;
    }
    void set_s2Inner(uint32_t val)
    {
        s2Inner = val;
    }
    void set_s2Tail(uint32_t val)
    {
        s2Tail = val;
    }
    void set_blockOuter(uint32_t val)
    {
        blockOuter = val;
    }
    void set_maxValidBBLen(uint32_t val)
    {
        maxValidBBLen = val;
    }
    void set_bandIdx(int64_t val)
    {
        bandIdx = val;
    }
};

class BlockNumListParamsRegbase {
public:
    int64_t blockStarts[MAX_CORE_NUM];
    int64_t blockEnds[MAX_CORE_NUM];

    const int64_t *get_blockStarts() const
    {
        return blockStarts;
    }
    int64_t get_blockStarts(int index) const
    {
        return blockStarts[index];
    }
    const int64_t *get_blockEnds() const
    {
        return blockEnds;
    }
    int64_t get_blockEnds(int index) const
    {
        return blockEnds[index];
    }
    void set_blockStarts(const int64_t *val)
    {
        for (uint i = 0; i < MAX_CORE_NUM; ++i) {
            blockStarts[i] = val[i];
        }
    }
    void set_blockStarts(int index, int64_t val)
    {
        blockStarts[index] = val;
    }
    void set_blockEnds(const int64_t *val)
    {
        for (uint i = 0; i < MAX_CORE_NUM; ++i) {
            blockEnds[i] = val[i];
        }
    }
    void set_blockEnds(int index, int64_t val)
    {
        blockEnds[index] = val;
    }
};

class PreParamsRegbase {
public:
    uint64_t maskSingleCoreNum;
    uint64_t qPreBlockFactor;
    uint64_t qPreBlockTotal;
    uint64_t qPreBlockTail;
    uint64_t kPreBlockFactor;
    uint64_t kPreBlockTotal;
    uint64_t kPreBlockTail;
    uint64_t vPreBlockFactor;
    uint64_t vPreBlockTotal;
    uint64_t vPreBlockTail;
    uint64_t sinkPreBlockFactor;
    uint64_t sinkPreBlockTotal;
    uint64_t sinkPreBlockTail;
    uint64_t maskCoreNum;
    uint64_t castBufferLen;
    uint64_t outputBufferLen;
    uint64_t inputBufferLen;
    uint64_t singleUBProcessNum;
    uint64_t maskSingleCoreLoop;
    uint64_t maskLastLoopNum;
    uint64_t maskTailCoreLoop;
    uint64_t maskTailCoreLastLoopNum;
    uint32_t dropoutIsDivisibleBy8;
    bool sValueZeroUnderTND;
    bool hasInvalidCol;
    uint8_t reserved2; // tilingData需要8字节对齐
    uint8_t reserved3; // tilingData需要8字节对齐

    // softmax grad
    uint32_t sfmgUsedCoreNum;
    uint32_t sfmgDyBufferLen;
    uint32_t sfmgYBufferLen;
    uint32_t sfmgOutputBufferLen;
    uint64_t singleLoopNBurstNum;
    uint64_t normalCoreLoopTimes;
    uint64_t tailCoreLoopTimes;
    uint64_t normalCoreLastLoopNBurstNum;
    uint64_t tailCoreLastLoopNBurstNum;
    uint64_t normalCoreNBurstNums;
    uint64_t tailCoreNBurstNums;
    uint64_t normalAxisSize;

    uint64_t get_maskSingleCoreNum() const
    {
        return maskSingleCoreNum;
    }
    uint64_t get_qPreBlockFactor() const
    {
        return qPreBlockFactor;
    }
    uint64_t get_qPreBlockTotal() const
    {
        return qPreBlockTotal;
    }
    uint64_t get_qPreBlockTail() const
    {
        return qPreBlockTail;
    }
    uint64_t get_kPreBlockFactor() const
    {
        return kPreBlockFactor;
    }
    uint64_t get_kPreBlockTotal() const
    {
        return kPreBlockTotal;
    }
    uint64_t get_kPreBlockTail() const
    {
        return kPreBlockTail;
    }
    uint64_t get_vPreBlockFactor() const
    {
        return vPreBlockFactor;
    }
    uint64_t get_vPreBlockTotal() const
    {
        return vPreBlockTotal;
    }
    uint64_t get_vPreBlockTail() const
    {
        return vPreBlockTail;
    }
    uint64_t get_sinkPreBlockFactor() const
    {
        return sinkPreBlockFactor;
    }
    uint64_t get_sinkPreBlockTotal() const
    {
        return sinkPreBlockTotal;
    }
    uint64_t get_sinkPreBlockTail() const
    {
        return sinkPreBlockTail;
    }
    uint64_t get_maskCoreNum() const
    {
        return maskCoreNum;
    }
    uint64_t get_castBufferLen() const
    {
        return castBufferLen;
    }
    uint64_t get_outputBufferLen() const
    {
        return outputBufferLen;
    }
    uint64_t get_inputBufferLen() const
    {
        return inputBufferLen;
    }
    uint64_t get_singleUBProcessNum() const
    {
        return singleUBProcessNum;
    }
    uint64_t get_maskSingleCoreLoop() const
    {
        return maskSingleCoreLoop;
    }
    uint64_t get_maskLastLoopNum() const
    {
        return maskLastLoopNum;
    }
    uint64_t get_maskTailCoreLoop() const
    {
        return maskTailCoreLoop;
    }
    uint64_t get_maskTailCoreLastLoopNum() const
    {
        return maskTailCoreLastLoopNum;
    }
    uint32_t get_dropoutIsDivisibleBy8() const
    {
        return dropoutIsDivisibleBy8;
    }
    bool get_sValueZeroUnderTND() const
    {
        return sValueZeroUnderTND;
    }
    bool get_hasInvalidCol() const
    {
        return hasInvalidCol;
    }
    uint32_t get_sfmgUsedCoreNum() const
    {
        return sfmgUsedCoreNum;
    }
    uint32_t get_sfmgDyBufferLen() const
    {
        return sfmgDyBufferLen;
    }
    uint32_t get_sfmgYBufferLen() const
    {
        return sfmgYBufferLen;
    }
    uint32_t get_sfmgOutputBufferLen() const
    {
        return sfmgOutputBufferLen;
    }
    uint64_t get_singleLoopNBurstNum() const
    {
        return singleLoopNBurstNum;
    }
    uint64_t get_normalCoreLoopTimes() const
    {
        return normalCoreLoopTimes;
    }
    uint64_t get_tailCoreLoopTimes() const
    {
        return tailCoreLoopTimes;
    }
    uint64_t get_normalCoreLastLoopNBurstNum() const
    {
        return normalCoreLastLoopNBurstNum;
    }
    uint64_t get_tailCoreLastLoopNBurstNum() const
    {
        return tailCoreLastLoopNBurstNum;
    }
    uint64_t get_normalCoreNBurstNums() const
    {
        return normalCoreNBurstNums;
    }
    uint64_t get_tailCoreNBurstNums() const
    {
        return tailCoreNBurstNums;
    }
    uint64_t get_normalAxisSize() const
    {
        return normalAxisSize;
    }

    void set_maskSingleCoreNum(uint64_t val)
    {
        maskSingleCoreNum = val;
    }
    void set_qPreBlockFactor(uint64_t val)
    {
        qPreBlockFactor = val;
    }
    void set_qPreBlockTotal(uint64_t val)
    {
        qPreBlockTotal = val;
    }
    void set_qPreBlockTail(uint64_t val)
    {
        qPreBlockTail = val;
    }
    void set_kPreBlockFactor(uint64_t val)
    {
        kPreBlockFactor = val;
    }
    void set_kPreBlockTotal(uint64_t val)
    {
        kPreBlockTotal = val;
    }
    void set_kPreBlockTail(uint64_t val)
    {
        kPreBlockTail = val;
    }
    void set_vPreBlockFactor(uint64_t val)
    {
        vPreBlockFactor = val;
    }
    void set_vPreBlockTotal(uint64_t val)
    {
        vPreBlockTotal = val;
    }
    void set_vPreBlockTail(uint64_t val)
    {
        vPreBlockTail = val;
    }
    void set_sinkPreBlockFactor(uint64_t val)
    {
        sinkPreBlockFactor = val;
    }
    void set_sinkPreBlockTotal(uint64_t val)
    {
        sinkPreBlockTotal = val;
    }
    void set_sinkPreBlockTail(uint64_t val)
    {
        sinkPreBlockTail = val;
    }
    void set_maskCoreNum(uint64_t val)
    {
        maskCoreNum = val;
    }
    void set_castBufferLen(uint64_t val)
    {
        castBufferLen = val;
    }
    void set_outputBufferLen(uint64_t val)
    {
        outputBufferLen = val;
    }
    void set_inputBufferLen(uint64_t val)
    {
        inputBufferLen = val;
    }
    void set_singleUBProcessNum(uint64_t val)
    {
        singleUBProcessNum = val;
    }
    void set_maskSingleCoreLoop(uint64_t val)
    {
        maskSingleCoreLoop = val;
    }
    void set_maskLastLoopNum(uint64_t val)
    {
        maskLastLoopNum = val;
    }
    void set_maskTailCoreLoop(uint64_t val)
    {
        maskTailCoreLoop = val;
    }
    void set_maskTailCoreLastLoopNum(uint64_t val)
    {
        maskTailCoreLastLoopNum = val;
    }
    void set_dropoutIsDivisibleBy8(uint32_t val)
    {
        dropoutIsDivisibleBy8 = val;
    }
    void set_sValueZeroUnderTND(bool val)
    {
        sValueZeroUnderTND = val;
    }
    void set_hasInvalidCol(bool val)
    {
        hasInvalidCol = val;
    }
    void set_sfmgUsedCoreNum(uint32_t val)
    {
        sfmgUsedCoreNum = val;
    }
    void set_sfmgDyBufferLen(uint32_t val)
    {
        sfmgDyBufferLen = val;
    }
    void set_sfmgYBufferLen(uint32_t val)
    {
        sfmgYBufferLen = val;
    }
    void set_sfmgOutputBufferLen(uint32_t val)
    {
        sfmgOutputBufferLen = val;
    }
    void set_singleLoopNBurstNum(uint64_t val)
    {
        singleLoopNBurstNum = val;
    }
    void set_normalCoreLoopTimes(uint64_t val)
    {
        normalCoreLoopTimes = val;
    }
    void set_tailCoreLoopTimes(uint64_t val)
    {
        tailCoreLoopTimes = val;
    }
    void set_normalCoreLastLoopNBurstNum(uint64_t val)
    {
        normalCoreLastLoopNBurstNum = val;
    }
    void set_tailCoreLastLoopNBurstNum(uint64_t val)
    {
        tailCoreLastLoopNBurstNum = val;
    }
    void set_normalCoreNBurstNums(uint64_t val)
    {
        normalCoreNBurstNums = val;
    }
    void set_tailCoreNBurstNums(uint64_t val)
    {
        tailCoreNBurstNums = val;
    }
    void set_normalAxisSize(uint64_t val)
    {
        normalAxisSize = val;
    }
};

class PostParamsRegbase {
public:
    uint64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    uint64_t qPostBaseNum;
    uint64_t qPostTailNum;
    uint64_t kPostBlockFactor;
    uint64_t kPostBlockTotal;
    uint64_t kPostBaseNum;
    uint64_t kPostTailNum;
    uint64_t vPostBlockFactor;
    uint64_t vPostBlockTotal;
    uint64_t vPostBaseNum;
    uint64_t vPostTailNum;
    uint64_t dqWorkSpaceOffset;
    uint64_t dkWorkSpaceOffset;
    uint64_t dvWorkSpaceOffset;
    uint64_t dropMaskGmOffset;
    uint64_t sfmgWorkSpaceOffset;
    uint64_t dsinkWorkSpaceOffset;
    uint64_t sinkReduceAxis;
    uint64_t sinkPostBlockTotal;
    uint64_t sinkPostBlockFactor;
    uint64_t sinkPostTailNum;

    uint64_t get_qPostBlockFactor() const
    {
        return qPostBlockFactor;
    }
    uint64_t get_qPostBlockTotal() const
    {
        return qPostBlockTotal;
    }
    uint64_t get_qPostBaseNum() const
    {
        return qPostBaseNum;
    }
    uint64_t get_qPostTailNum() const
    {
        return qPostTailNum;
    }
    uint64_t get_kPostBlockFactor() const
    {
        return kPostBlockFactor;
    }
    uint64_t get_kPostBlockTotal() const
    {
        return kPostBlockTotal;
    }
    uint64_t get_kPostBaseNum() const
    {
        return kPostBaseNum;
    }
    uint64_t get_kPostTailNum() const
    {
        return kPostTailNum;
    }
    uint64_t get_vPostBlockFactor() const
    {
        return vPostBlockFactor;
    }
    uint64_t get_vPostBlockTotal() const
    {
        return vPostBlockTotal;
    }
    uint64_t get_vPostBaseNum() const
    {
        return vPostBaseNum;
    }
    uint64_t get_vPostTailNum() const
    {
        return vPostTailNum;
    }
    uint64_t get_dqWorkSpaceOffset() const
    {
        return dqWorkSpaceOffset;
    }
    uint64_t get_dkWorkSpaceOffset() const
    {
        return dkWorkSpaceOffset;
    }
    uint64_t get_dvWorkSpaceOffset() const
    {
        return dvWorkSpaceOffset;
    }
    uint64_t get_dropMaskGmOffset() const
    {
        return dropMaskGmOffset;
    }
    uint64_t get_sfmgWorkSpaceOffset() const
    {
        return sfmgWorkSpaceOffset;
    }
    uint64_t get_dsinkWorkSpaceOffset() const
    {
        return dsinkWorkSpaceOffset;
    }
    uint64_t get_sinkReduceAxis() const
    {
        return sinkReduceAxis;
    }
    uint64_t get_sinkPostBlockTotal() const
    {
        return sinkPostBlockTotal;
    }
    uint64_t get_sinkPostBlockFactor() const
    {
        return sinkPostBlockFactor;
    }
    uint64_t get_sinkPostTailNum() const
    {
        return sinkPostTailNum;
    }

    void set_qPostBlockFactor(uint64_t value)
    {
        qPostBlockFactor = value;
    }
    void set_qPostBlockTotal(uint64_t value)
    {
        qPostBlockTotal = value;
    }
    void set_qPostBaseNum(uint64_t value)
    {
        qPostBaseNum = value;
    }
    void set_qPostTailNum(uint64_t value)
    {
        qPostTailNum = value;
    }
    void set_kPostBlockFactor(uint64_t value)
    {
        kPostBlockFactor = value;
    }
    void set_kPostBlockTotal(uint64_t value)
    {
        kPostBlockTotal = value;
    }
    void set_kPostBaseNum(uint64_t value)
    {
        kPostBaseNum = value;
    }
    void set_kPostTailNum(uint64_t value)
    {
        kPostTailNum = value;
    }
    void set_vPostBlockFactor(uint64_t value)
    {
        vPostBlockFactor = value;
    }
    void set_vPostBlockTotal(uint64_t value)
    {
        vPostBlockTotal = value;
    }
    void set_vPostBaseNum(uint64_t value)
    {
        vPostBaseNum = value;
    }
    void set_vPostTailNum(uint64_t value)
    {
        vPostTailNum = value;
    }
    void set_dqWorkSpaceOffset(uint64_t value)
    {
        dqWorkSpaceOffset = value;
    }
    void set_dkWorkSpaceOffset(uint64_t value)
    {
        dkWorkSpaceOffset = value;
    }
    void set_dvWorkSpaceOffset(uint64_t value)
    {
        dvWorkSpaceOffset = value;
    }
    void set_dropMaskGmOffset(uint64_t value)
    {
        dropMaskGmOffset = value;
    }
    void set_sfmgWorkSpaceOffset(uint64_t value)
    {
        sfmgWorkSpaceOffset = value;
    }
    void set_dsinkWorkSpaceOffset(uint64_t value)
    {
        dsinkWorkSpaceOffset = value;
    }
    void set_sinkReduceAxis(uint64_t value)
    {
        sinkReduceAxis = value;
    }
    void set_sinkPostBlockTotal(uint64_t value)
    {
        sinkPostBlockTotal = value;
    }
    void set_sinkPostBlockFactor(uint64_t value)
    {
        sinkPostBlockFactor = value;
    }
    void set_sinkPostTailNum(uint64_t value)
    {
        sinkPostTailNum = value;
    }
};

class BaseDeterParamRegbase {
public:
    uint32_t noNeedDeter;
    uint32_t reserved1; // tilingData需要8字节对齐
    int64_t deterMaxRound;
    uint64_t deterGmOffset;
    uint64_t deterWorkSpaceOffset;
    uint64_t dqIsNeedDeter[MAX_CORE_NUM];
    uint64_t dkDvIsNeedDeter[MAX_CORE_NUM];

    uint32_t get_noNeedDeter() const
    {
        return noNeedDeter;
    }
    int64_t get_deterMaxRound() const
    {
        return deterMaxRound;
    }
    uint64_t get_deterGmOffset() const
    {
        return deterGmOffset;
    }
    uint64_t get_deterWorkSpaceOffset() const
    {
        return deterWorkSpaceOffset;
    }
    uint64_t get_dqIsNeedDeter(int index) const
    {
        return dqIsNeedDeter[index];
    }
    const uint64_t *get_dqIsNeedDeter() const
    {
        return dqIsNeedDeter;
    }
    const uint64_t *get_dkDvIsNeedDeter() const
    {
        return dkDvIsNeedDeter;
    }
    uint64_t get_dkDvIsNeedDeter(int index) const
    {
        return dkDvIsNeedDeter[index];
    }

    void set_noNeedDeter(uint32_t val)
    {
        noNeedDeter = val;
    }
    void set_deterMaxRound(int64_t value)
    {
        deterMaxRound = value;
    }
    void set_deterGmOffset(uint64_t value)
    {
        deterGmOffset = value;
    }
    void set_deterWorkSpaceOffset(uint64_t value)
    {
        deterWorkSpaceOffset = value;
    }
    void set_dqIsNeedDeter(const uint64_t *val)
    {
        for (uint i = 0; i < MAX_CORE_NUM; ++i) {
            dqIsNeedDeter[i] = val[i];
        }
    }
    void set_dqIsNeedDeter(int index, uint64_t val)
    {
        dqIsNeedDeter[index] = val;
    }
    void set_dkDvIsNeedDeter(const uint64_t *val)
    {
        for (uint i = 0; i < MAX_CORE_NUM; ++i) {
            dkDvIsNeedDeter[i] = val[i];
        }
    }
    void set_dkDvIsNeedDeter(int index, uint64_t val)
    {
        dkDvIsNeedDeter[index] = val;
    }
};

class DeterParamRegbase {
public:
    constexpr static int64_t DETER_PREFIX_NUM = 132;

    bool coreDivide;
    uint8_t tndLineDeter;
    uint8_t reserved2;  // tilingData需要8字节对齐
    uint8_t reserved3;  // tilingData需要8字节对齐
    uint32_t reserved4; // tilingData需要8字节对齐
    int64_t deterPrefixStep;
    int64_t deterPrefix[DETER_PREFIX_NUM];
    int64_t deterPrefixAlign[DETER_PREFIX_NUM];
    int64_t deterPrefix0[DETER_PREFIX_NUM];
    int64_t deterPrefix1[DETER_PREFIX_NUM];
    int64_t deterPrefix2[DETER_PREFIX_NUM];

    bool get_coreDivide() const
    {
        return coreDivide;
    }
    uint8_t get_tndLineDeter() const
    {
        return tndLineDeter;
    }
    int64_t get_deterPrefixStep() const
    {
        return deterPrefixStep;
    }
    const int64_t *get_deterPrefix() const
    {
        return deterPrefix;
    }
    int64_t get_deterPrefix(int index) const
    {
        return deterPrefix[index];
    }
    const int64_t *get_deterPrefixAlign() const
    {
        return deterPrefixAlign;
    }
    int64_t get_deterPrefixAlign(int index) const
    {
        return deterPrefixAlign[index];
    }
    const int64_t *get_deterPrefix0() const
    {
        return deterPrefix0;
    }
    int64_t get_deterPrefix0(int index) const
    {
        return deterPrefix0[index];
    }
    const int64_t *get_deterPrefix1() const
    {
        return deterPrefix1;
    }
    int64_t get_deterPrefix1(int index) const
    {
        return deterPrefix1[index];
    }
    const int64_t *get_deterPrefix2() const
    {
        return deterPrefix2;
    }
    int64_t get_deterPrefix2(int index) const
    {
        return deterPrefix2[index];
    }
    void set_coreDivide(bool value)
    {
        coreDivide = value;
    }
    void set_tndLineDeter(uint8_t value)
    {
        tndLineDeter = value;
    }
    void set_deterPrefixStep(int64_t value)
    {
        deterPrefixStep = value;
    }
    void set_deterPrefix(const int64_t *val)
    {
        for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
            deterPrefix[i] = val[i];
        }
    }
    void set_deterPrefix(int index, int64_t val)
    {
        deterPrefix[index] = val;
    }
    void set_deterPrefixAlign(const int64_t *val)
    {
        for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
            deterPrefixAlign[i] = val[i];
        }
    }
    void set_deterPrefixAlign(int index, int64_t val)
    {
        deterPrefixAlign[index] = val;
    }
    void set_deterPrefix0(const int64_t *val)
    {
        for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
            deterPrefix0[i] = val[i];
        }
    }
    void set_deterPrefix0(int index, int64_t val)
    {
        deterPrefix0[index] = val;
    }
    void set_deterPrefix1(const int64_t *val)
    {
        for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
            deterPrefix1[i] = val[i];
        }
    }
    void set_deterPrefix1(int index, int64_t val)
    {
        deterPrefix1[index] = val;
    }
    void set_deterPrefix2(const int64_t *val)
    {
        for (int i = 0; i < DETER_PREFIX_NUM; ++i) {
            deterPrefix2[i] = val[i];
        }
    }
    void set_deterPrefix2(int index, int64_t val)
    {
        deterPrefix2[index] = val;
    }
};

class TndParamRegbase {
public:
    uint64_t tndStartBIdx[MAX_CORE_NUM];
    uint64_t tndS1S2PrefixSum[MAX_CORE_NUM];
    uint64_t tndS1S2AlignPrefixSum[MAX_CORE_NUM];
    uint64_t tndPrefixSum[MAX_CORE_NUM];
    void set_tndStartBIdx(const uint64_t *val)
    {
        for (uint i = 0; i < MAX_CORE_NUM; ++i) {
            tndStartBIdx[i] = val[i];
        }
    }
    void set_tndS1S2PrefixSum(const uint64_t *val)
    {
        for (uint i = 0; i < MAX_CORE_NUM; ++i) {
            tndS1S2PrefixSum[i] = val[i];
        }
    }
    void set_tndS1S2AlignPrefixSum(const uint64_t *val)
    {
        for (uint i = 0; i < MAX_CORE_NUM; ++i) {
            tndS1S2AlignPrefixSum[i] = val[i];
        }
    }
    void set_tndPrefixSum(const uint64_t *val)
    {
        for (uint i = 0; i < MAX_CORE_NUM; ++i) {
            tndPrefixSum[i] = val[i];
        }
    }
};

class TndSwizzleParamRegbase {
public:
    uint64_t tndSwizzleS1S2PrefixSum[TND_SWIZZLE_PREFIX_NUM];
    uint64_t tndSwizzleS1S2AlignPrefixSum[TND_SWIZZLE_PREFIX_NUM];
    uint64_t tndS2BlockPrefixSum[TND_SWIZZLE_PREFIX_NUM];

    void set_tndSwizzleS1S2PrefixSum(const uint64_t *val)
    {
        for (int i = 0; i < TND_SWIZZLE_PREFIX_NUM; ++i) {
            tndSwizzleS1S2PrefixSum[i] = val[i];
        }
    }
    void set_tndSwizzleS1S2AlignPrefixSum(const uint64_t *val)
    {
        for (int i = 0; i < TND_SWIZZLE_PREFIX_NUM; ++i) {
            tndSwizzleS1S2AlignPrefixSum[i] = val[i];
        }
    }
    void set_tndS2BlockPrefixSum(const uint64_t *val)
    {
        for (int i = 0; i < TND_SWIZZLE_PREFIX_NUM; ++i) {
            tndS2BlockPrefixSum[i] = val[i];
        }
    }
};

template <const bool isDeter = false, const bool isNewDeter = false, const bool isTnd = false,
          const bool isTndSwizzle = false>
class FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase {
public:
    FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase s1s2BNGS1S2BaseParams;
    FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsRegbase s1s2BNGS1S2SplitCoreParams;
    BlockNumListParamsRegbase s1s2BNGS1S2BlockNumList;
    PreParamsRegbase preTilingData;
    PostParamsRegbase postTilingData;
    typename std::conditional<isDeter, BaseDeterParamRegbase, std::nullptr_t>::type baseDeterParam;
    typename std::conditional<isNewDeter, DeterParamRegbase, std::nullptr_t>::type deterParam;
    typename std::conditional<!isNewDeter && isTnd, TndParamRegbase, std::nullptr_t>::type tndParam;
    typename std::conditional<isTndSwizzle, TndSwizzleParamRegbase, std::nullptr_t>::type tndSwizzleParam;
};
} // namespace fag
} // namespace optiling
#endif
